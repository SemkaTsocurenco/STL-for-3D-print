import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# ----------------------------------------------------------------
# 1. ЗАГРУЗКА STL-МОДЕЛИ
# ----------------------------------------------------------------
def load_stl_model(filename: str) -> mesh.Mesh:
    """Загружает STL-модель (ASCII или binary)."""
    return mesh.Mesh.from_file(filename)

# ----------------------------------------------------------------
# 2. БЫСТРАЯ ВИЗУАЛИЗАЦИЯ (ОБЛАКО ТОЧЕК STL)
# ----------------------------------------------------------------
def plot_stl_points(ax, model: mesh.Mesh, alpha=0.2):
    """
    Рисует облако вершин (x,y,z) для всей STL,
    чтобы не перегружать matplotlib тысячами линий.
    Возвращает объект PathCollection (scatter),
    чтобы управлять его видимостью.
    """
    pts = model.points.reshape(-1, 3)
    xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
    scatter_obj = ax.scatter(xs, ys, zs, s=1, c='gray', alpha=alpha)
    return scatter_obj

# ----------------------------------------------------------------
# ФУНКЦИЯ ДЛЯ ОТОБРАЖЕНИЯ МОДЕЛИ В ВИДЕ ЗАКРАШЕННЫХ ПОЛИГОНОВ
# ----------------------------------------------------------------
def plot_stl_mesh(ax, model: mesh.Mesh, color='blue', alpha=0.3):
    """
    Рисует STL-модель в виде закрашенных полупрозрачных полигонов.
    Возвращает объект Poly3DCollection, чтобы управлять видимостью.
    """
    faces = model.vectors  # shape (N,3,3)
    poly3d = [face for face in faces]
    mesh_collection = Poly3DCollection(poly3d, facecolors=color, alpha=alpha, edgecolors='k', linewidths=0.2)
    ax.add_collection3d(mesh_collection)
    return mesh_collection

# ----------------------------------------------------------------
# 3. ПЕРЕСЕЧЕНИЕ ТРЕУГОЛЬНИКА С ПЛОСКОСТЬЮ Z = z_plane
# ----------------------------------------------------------------
def triangle_plane_intersection(tri_coords: np.ndarray, z_plane: float):
    """
    Возвращает 0..2 точки пересечения треугольника (3D) с плоскостью z = z_plane.
    Каждый tri_coords – это shape (3,3), три вершины в 3D.
    """
    points = []
    z_vals = tri_coords[:, 2]
    above = z_vals >= z_plane
    below = z_vals < z_plane

    if np.all(above) or np.all(below):
        return []

    for i in range(3):
        j = (i+1) % 3
        z_i = z_vals[i]
        z_j = z_vals[j]
        if (z_i < z_plane <= z_j) or (z_j < z_plane <= z_i):
            t = (z_plane - z_i) / (z_j - z_i)
            p = tri_coords[i] + t*(tri_coords[j] - tri_coords[i])
            points.append(p)
    return points

# ----------------------------------------------------------------
# 4. РАЗРЕЗАЕМ МОДЕЛЬ НА СЛОИ
# ----------------------------------------------------------------
def slice_model(model: mesh.Mesh, layer_height: float):
    z_min, z_max = model.z.min(), model.z.max()
    tri_array = model.vectors

    layers = []
    z_current = z_min
    while z_current <= (z_max + 1e-9):
        segs = []
        for tri in tri_array:
            pts = triangle_plane_intersection(tri, z_current)
            if len(pts) == 2:
                p1, p2 = pts
                segs.append( ((p1[0], p1[1], p1[2]),
                              (p2[0], p2[1], p2[2])) )
        layers.append((z_current, segs))
        z_current += layer_height
    return layers

# ----------------------------------------------------------------
# 5. ПОСТРОЕНИЕ КОНТУРОВ (ПОЛИЛИНИЙ) ИЗ НАБОРА СЕГМЕНТОВ НА СЛОЕ
# ----------------------------------------------------------------
def build_contours_from_segments(segments, eps=1e-7):
    """
    segments — список кортежей вида: ((x1,y1,z),(x2,y2,z)), ...
    Возвращаем список контуров, где каждый контур – список вершин (x,y,z),
    замкнутый (последняя точка совпадает с первой).
    """
    raw_points = []
    for seg in segments:
        raw_points.append(seg[0])
        raw_points.append(seg[1])
    raw_points = np.array(raw_points)

    used = np.zeros(len(raw_points), dtype=bool)
    unique_points = []
    map_index = [-1]*len(raw_points)

    def dist2(a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2

    for i in range(len(raw_points)):
        if used[i]:
            continue
        pi = raw_points[i]
        cluster = [i]
        used[i] = True
        for j in range(i+1, len(raw_points)):
            if not used[j]:
                pj = raw_points[j]
                if dist2(pi, pj) < eps*eps:
                    cluster.append(j)
                    used[j] = True
        cluster_coords = raw_points[cluster]
        avg_point = (
            np.mean(cluster_coords[:,0]),
            np.mean(cluster_coords[:,1]),
            np.mean(cluster_coords[:,2])
        )
        idx_new = len(unique_points)
        unique_points.append(avg_point)
        for c in cluster:
            map_index[c] = idx_new

    unique_points = np.array(unique_points)
    edges = []

    for seg in segments:
        p1, p2 = seg
        i1 = find_closest_index_in_raw(p1, raw_points, eps)
        i2 = find_closest_index_in_raw(p2, raw_points, eps)
        u = map_index[i1]
        v = map_index[i2]
        if u != v:
            edges.append((u,v))

    edges_uniq = set()
    for (a,b) in edges:
        if a < b:
            edges_uniq.add((a,b))
        else:
            edges_uniq.add((b,a))
    edges = list(edges_uniq)

    adjacency = {}
    for (a,b) in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    contours = []
    visited = set()

    def walk_chain(start):
        chain = [start]
        current = start
        visited.add(current)
        while True:
            neigh = adjacency[current]
            nxt = None
            for n in neigh:
                if n not in visited:
                    nxt = n
                    break
            if nxt is None:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        return chain

    for v in adjacency.keys():
        if v not in visited:
            chain = walk_chain(v)
            if chain[0] in adjacency[chain[-1]]:
                chain.append(chain[0])
            contours.append(chain)

    final_contours = []
    for c in contours:
        coords = [tuple(unique_points[idx]) for idx in c]
        final_contours.append(coords)

    return final_contours

def find_closest_index_in_raw(p, raw_points, eps):
    x, y, z = p
    d2 = (raw_points[:,0]-x)**2 + (raw_points[:,1]-y)**2 + (raw_points[:,2]-z)**2
    idx = np.argmin(d2)
    return idx

# ----------------------------------------------------------------
# 6. ПРОВЕРКА, ЛЕЖИТ ЛИ ТОЧКА ВНУТРИ ОДНОГО ЗАМКНУТОГО КОНТУРА (2D)
# ----------------------------------------------------------------
def point_in_polygon_2d(px, py, polygon):
    inside = False
    n = len(polygon)
    for i in range(n-1):
        x1, y1, _ = polygon[i]
        x2, y2, _ = polygon[i+1]
        if ((y1 > py) != (y2 > py)):
            x_int = x1 + (py - y1)*(x2-x1)/(y2-y1)
            if x_int > px:
                inside = not inside
    return inside

# ----------------------------------------------------------------
# 7. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ЗАПОЛНЕНИЯ
# ----------------------------------------------------------------
def generate_infill_for_layer(z_val, contours, nozzle_diameter, layer_index,
                              N_pts_line=200):
    """
    Генерируем линии "змейкой" (серпантином), чтобы
    не возвращаться холостым ходом назад в начало каждой строки.
    Возвращаем список точек (x,y,z,feed) с минимальным холостым перемещением.
    """
    if not contours:
        return []

    # Соберём bounding box
    all_xy = []
    for polygon in contours:
        for (x,y,z) in polygon:
            all_xy.append((x,y))
    arr_xy = np.array(all_xy)
    x_min, x_max = arr_xy[:,0].min(), arr_xy[:,0].max()
    y_min, y_max = arr_xy[:,1].min(), arr_xy[:,1].max()

    line_spacing = nozzle_diameter
    all_segments_in_layer = []

    # Выбираем ориентацию слоя "крест-накрест" (по признаку layer_index)
    if layer_index % 2 == 0:
        # (x - y) = const
        start_D = x_min - y_max
        end_D   = x_max - y_min
        D = start_D
        line_count = 0
        while D <= end_D + 1e-9:
            # для чётных строк идём "вперёд", для нечётных - "назад"
            if line_count % 2 == 0:
                ys = np.linspace(y_min, y_max, N_pts_line)
            else:
                ys = np.linspace(y_max, y_min, N_pts_line)
            line_points = []
            for y in ys:
                x = y + D
                line_points.append((x,y,z_val))
            clipped_segments = clip_line_by_polygons(line_points, contours)
            all_segments_in_layer.extend(clipped_segments)
            D += line_spacing
            line_count += 1
    else:
        # (x + y) = const
        start_D = x_min + y_min
        end_D   = x_max + y_max
        D = start_D
        line_count = 0
        while D <= end_D + 1e-9:
            if line_count % 2 == 0:
                ys = np.linspace(y_min, y_max, N_pts_line)
            else:
                ys = np.linspace(y_max, y_min, N_pts_line)
            line_points = []
            for y in ys:
                x = D - y
                line_points.append((x,y,z_val))
            clipped_segments = clip_line_by_polygons(line_points, contours)
            all_segments_in_layer.extend(clipped_segments)
            D += line_spacing
            line_count += 1

    # Теперь объединяем все segments в единый "инструментальный" маршрут,
    # чтобы минимизировать холостой ход между соседними сегментами:
    # - первый сегмент начинается travel'ом в его первую точку
    # - далее печатаем
    # - между сегментами перемещаемся к началу следующего сегмента без подачи,
    #   НО без возврата куда-то в начало "по умолчанию" – идём от конца предыдущего к началу следующего.
    toolpath = []
    first_segment = True
    for seg in all_segments_in_layer:
        if len(seg) < 2:
            continue
        # Если это первый сегмент слоя:
        if first_segment:
            # Добавляем первую точку c feed=False (travel)
            toolpath.append((*seg[0], False))
            first_segment = False
        else:
            # Смотрим, где мы закончили предыдущий сегмент
            x_last, y_last, z_last, feed_last = toolpath[-1]
            # Если начало нового сегмента не совпадает с концом предыдущего — travel:
            if (abs(x_last - seg[0][0])>1e-9 or
                abs(y_last - seg[0][1])>1e-9 or
                abs(z_last - seg[0][2])>1e-9):
                toolpath.append((*seg[0], False))

        # Теперь все точки сегмента печатаем с подачей = True (кроме самой первой, если уже есть)
        for i, pt in enumerate(seg):
            if i == 0:
                # уже добавлена выше (travel) — поставили feed=False
                continue
            toolpath.append((*pt, True))

    return toolpath

def clip_line_by_polygons(line_points, polygons):
    inside_flags = []
    for p in line_points:
        x,y,z = p
        if is_inside_any_polygon_2d(x, y, polygons):
            inside_flags.append(True)
        else:
            inside_flags.append(False)

    segments = []
    current_segment = []
    for i, flag in enumerate(inside_flags):
        if flag:
            current_segment.append(line_points[i])
        else:
            if len(current_segment) > 0:
                segments.append(current_segment)
                current_segment = []
    if len(current_segment) > 0:
        segments.append(current_segment)
    return segments

def is_inside_any_polygon_2d(x, y, polygons):
    for poly in polygons:
        if point_in_polygon_2d(x, y, poly):
            return True
    return False

# ----------------------------------------------------------------
# 8. СТРОИМ ПОЛНУЮ ТРАЕКТОРИЮ ПО ВСЕМ СЛОЯМ
# ----------------------------------------------------------------
def build_full_toolpath(layers, nozzle_diameter=0.4):
    """
    Здесь мы тоже не возвращаемся в начало слоя:
      каждый слой строится "змейкой" в generate_infill_for_layer,
      а между слоями идём travel от конца предыдущего слоя к началу следующего.
    """
    full_path = []
    prev_end = None

    for i, (z_val, segs) in enumerate(layers):
        contours = build_contours_from_segments(segs)
        if not contours:
            continue

        layer_path = generate_infill_for_layer(z_val, contours,
                                               nozzle_diameter,
                                               layer_index=i,
                                               N_pts_line=200)
        if not layer_path:
            continue

        # travel между слоями
        if prev_end is not None:
            xA, yA, zA, fA = prev_end
            xB, yB, zB, fB = layer_path[0]
            # если первая точка не совпадает с концом предыдущего слоя,
            # делаем travel
            if (abs(xA - xB)>1e-9 or abs(yA - yB)>1e-9 or abs(zA - zB)>1e-9):
                full_path.append((xA, yA, zA, False))
                full_path.append((xB, yB, zB, False))

        full_path.extend(layer_path)
        prev_end = layer_path[-1]

    return full_path

# ----------------------------------------------------------------
# 9. ПРЕОБРАЗОВАНИЕ ПУТИ В СЕГМЕНТЫ ДЛЯ ОТОБРАЖЕНИЯ
# ----------------------------------------------------------------
def toolpath_to_segments(toolpath):
    feed_segments = []
    travel_segments = []
    for i in range(len(toolpath)-1):
        x1, y1, z1, f1 = toolpath[i]
        x2, y2, z2, f2 = toolpath[i+1]
        seg = ((x1,y1,z1), (x2,y2,z2))
        if f2:
            feed_segments.append(seg)
        else:
            travel_segments.append(seg)
    return feed_segments, travel_segments

# ----------------------------------------------------------------
# 10. УТИЛИТА ДЛЯ БЫСТРОГО СОЗДАНИЯ Line3DCollection
# ----------------------------------------------------------------
def make_3dcollection(segments, color='red'):
    seg_array = []
    for s in segments:
        seg_array.append([
            [s[0][0], s[0][1], s[0][2]],
            [s[1][0], s[1][1], s[1][2]]
        ])
    seg_array = np.array(seg_array)
    lc = Line3DCollection(seg_array, colors=color, linewidths=1)
    return lc

# ----------------------------------------------------------------
# ГЕНЕРАЦИЯ 3D-ПРЯМОУГОЛЬНИКОВ (ПЛАНАРНЫХ) ДЛЯ ВИЗУАЛИЗАЦИИ СЛОЁВ
# ----------------------------------------------------------------
def build_layer_planes(layers, x_min, x_max, y_min, y_max, color='blue', alpha=0.1):
    """
    Создаёт Poly3DCollection из прямоугольных плоскостей
    (x_min,y_min,z), (x_min,y_max,z), (x_max,y_max,z), (x_max,y_min,z)
    для каждого слоя.
    """
    plane_polys = []
    for (z_val, _) in layers:
        plane_polys.append([
            (x_min, y_min, z_val),
            (x_min, y_max, z_val),
            (x_max, y_max, z_val),
            (x_max, y_min, z_val)
        ])
    collection = Poly3DCollection(plane_polys, facecolors=color, alpha=alpha, edgecolors='none')
    return collection

# ----------------------------------------------------------------
# ПОСТРОЕНИЕ ФИОЛЕТОВЫХ КОНТУРОВ СРЕЗА
# ----------------------------------------------------------------
def build_slice_contours_collection(layers, color='purple'):
    """
    Проходим по всем слоям, строим контуры, 
    собираем их как набор 3D отрезков для Line3DCollection.
    """
    all_contour_segs = []
    for z_val, segs in layers:
        contours = build_contours_from_segments(segs)
        for contour in contours:
            for i in range(len(contour)-1):
                all_contour_segs.append((contour[i], contour[i+1]))
    return make_3dcollection(all_contour_segs, color=color)

# ----------------------------------------------------------------
# 12. ГЛАВНАЯ ФУНКЦИЯ
# ----------------------------------------------------------------
def main():
    stl_file = "Bishop_ascii.stl"
    model = load_stl_model(stl_file)

    # 1) Режем модель на слои
    layer_height = 0.005
    layers = slice_model(model, layer_height)

    # 2) Строим полную траекторию (с минимизированным холостым ходом)
    nozzle_diameter = 0.005
    full_path = build_full_toolpath(layers, nozzle_diameter)

    # 3) Разбиваем на feeding и travel segments
    feed_segs, travel_segs = toolpath_to_segments(full_path)

    # 4) Готовим 3D-окно
    fig = plt.figure(figsize=(10,8))
    ax3d = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.15, right=0.85)

    # 5) Рисуем облако точек STL (для наглядности)
    stl_scatter = plot_stl_points(ax3d, model, alpha=0.2)

    # 5.1) Рисуем модель в виде закрашенных полупрозрачных полигонов (синяя)
    stl_mesh = plot_stl_mesh(ax3d, model, color='blue', alpha=0.5)

    # 6) Добавляем линии для feed (красные) и travel (зелёные)
    feed_collection   = make_3dcollection(feed_segs, color='red')
    travel_collection = make_3dcollection(travel_segs, color='green')
    ax3d.add_collection3d(feed_collection)
    ax3d.add_collection3d(travel_collection)

    # 7) Фиолетовые контуры среза
    slice_contours_collection = build_slice_contours_collection(layers, color='purple')
    ax3d.add_collection3d(slice_contours_collection)

    # 8) Полупрозрачные плоскости (слои), тоже синие, но более прозрачные
    x_min, x_max = model.x.min(), model.x.max()
    y_min, y_max = model.y.min(), model.y.max()
    planes_collection = build_layer_planes(layers, x_min, x_max, y_min, y_max,
                                           color='blue', alpha=0.1)
    ax3d.add_collection3d(planes_collection)

    # 9) Настраиваем лимиты осей
    xs, ys, zs = model.x, model.y, model.z
    ax3d.set_xlim(xs.min(), xs.max())
    ax3d.set_ylim(ys.min(), ys.max())
    ax3d.set_zlim(zs.min(), zs.max())
    ax3d.set_title("Cross-Cross Infill WITH Clipping by Model Cross-Section (Minimized Travel)")

    # 10) Добавим CheckButtons для управления видимостью
    ax_check = plt.axes([0.88, 0.3, 0.1, 0.35])
    labels = ["Points", "Mesh", "Feed", "Travel", "Contours", "Planes"]
    visibility = [True, True, True, True, True, True]
    check_btns = CheckButtons(ax_check, labels, visibility)

    def on_check(label):
        if label == "Points":
            stl_scatter.set_visible(not stl_scatter.get_visible())
        elif label == "Mesh":
            stl_mesh.set_visible(not stl_mesh.get_visible())
        elif label == "Feed":
            feed_collection.set_visible(not feed_collection.get_visible())
        elif label == "Travel":
            travel_collection.set_visible(not travel_collection.get_visible())
        elif label == "Contours":
            slice_contours_collection.set_visible(not slice_contours_collection.get_visible())
        elif label == "Planes":
            planes_collection.set_visible(not planes_collection.get_visible())
        plt.draw()

    check_btns.on_clicked(on_check)

    plt.show()

if __name__ == "__main__":
    main()

