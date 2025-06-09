from flask import Flask, render_template, request, jsonify
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

app = Flask(__name__)

class PSO:
    def __init__(self, n_particles, n_clusters, data, max_iter=100):
        self.n_particles = n_particles
        self.n_clusters = n_clusters
        self.data = data
        self.max_iter = max_iter
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive weight
        self.c2 = 1.5  # Social weight
        
        # Khởi tạo các particle
        self.particles = []
        self.velocities = []
        self.pbest = []
        self.pbest_scores = []
        self.gbest = None
        self.gbest_score = float('inf')
        
        # Khởi tạo ngẫu nhiên các particle
        for _ in range(n_particles):
            # Khởi tạo tâm cụm ngẫu nhiên
            centers = np.random.choice(data.flatten(), size=n_clusters, replace=False)
            centers = centers.reshape(-1, 1)  # Reshape để phù hợp với dữ liệu
            self.particles.append(centers)
            self.velocities.append(np.zeros_like(centers))
            self.pbest.append(centers.copy())
            score = self._calculate_fitness(centers)
            self.pbest_scores.append(score)
            if score < self.gbest_score:
                self.gbest = centers.copy()
                self.gbest_score = score
    
    def _calculate_fitness(self, centers):
        """Tính toán hàm mục tiêu (tổng khoảng cách bình phương)"""
        # Tính khoảng cách từ mỗi điểm đến mỗi tâm cụm
        distances = np.zeros((len(self.data), self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sum((self.data - centers[i]) ** 2, axis=1)
        
        # Gán mỗi điểm vào cụm gần nhất
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Tính tổng khoảng cách bình phương
        total_distance = 0
        for i in range(self.n_clusters):
            cluster_points = self.data[cluster_assignments == i]
            if len(cluster_points) > 0:
                total_distance += np.sum((cluster_points - centers[i]) ** 2)
        
        return total_distance
    
    def _update_velocity(self, particle_idx):
        """Cập nhật vận tốc của particle"""
        r1, r2 = np.random.rand(2)
        cognitive = self.c1 * r1 * (self.pbest[particle_idx] - self.particles[particle_idx])
        social = self.c2 * r2 * (self.gbest - self.particles[particle_idx])
        self.velocities[particle_idx] = (self.w * self.velocities[particle_idx] + 
                                       cognitive + social)
    
    def _update_position(self, particle_idx):
        """Cập nhật vị trí của particle"""
        self.particles[particle_idx] += self.velocities[particle_idx]
    
    def optimize(self):
        """Thực hiện tối ưu hóa PSO"""
        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                # Cập nhật vận tốc và vị trí
                self._update_velocity(i)
                self._update_position(i)
                
                # Tính toán fitness mới
                score = self._update_fitness(i)
                
                # Cập nhật pbest và gbest
                if score < self.pbest_scores[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_scores[i] = score
                    if score < self.gbest_score:
                        self.gbest = self.particles[i].copy()
                        self.gbest_score = score
        
        return self.gbest
    
    def _update_fitness(self, particle_idx):
        """Cập nhật fitness của particle"""
        return self._calculate_fitness(self.particles[particle_idx])

def hybrid_kmeans_pso(prices, k=7, n_particles=20, max_iter=100):
    """Bước 2: Áp dụng thuật toán K-means + PSO để chia tập EUA thành k khoảng có độ dài khác nhau
    
    Args:
        prices: Danh sách các giá trị cần chia khoảng
        k: Số khoảng cần chia (mặc định là 7)
        n_particles: Số lượng particle trong PSO
        max_iter: Số lần lặp tối đa
        
    Returns:
        List các tuple (start, end) đại diện cho các khoảng
    """
    # Chuẩn bị dữ liệu
    X = np.array(prices).reshape(-1, 1)
    
    # Bước 1: Thực hiện K-means ban đầu
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    initial_centers = kmeans.cluster_centers_
    
    # Bước 2: Sử dụng PSO để tối ưu hóa các tâm cụm
    pso = PSO(n_particles=n_particles, n_clusters=k, data=X, max_iter=max_iter)
    optimized_centers = pso.optimize()
    
    # Sắp xếp các tâm cụm
    optimized_centers = np.sort(optimized_centers.flatten())
    
    # Tính toán các khoảng dựa trên tâm cụm tối ưu
    intervals = []
    for i in range(len(optimized_centers)-1):
        mid_point = (optimized_centers[i] + optimized_centers[i+1]) / 2
        intervals.append((optimized_centers[i], mid_point))
    
    # Thêm khoảng đầu và cuối
    first_interval = (optimized_centers[0] - (optimized_centers[1] - optimized_centers[0]), 
                     optimized_centers[0])
    last_interval = (optimized_centers[-1], 
                    optimized_centers[-1] + (optimized_centers[-1] - optimized_centers[-2]))
    intervals.insert(0, first_interval)
    intervals.append(last_interval)
    
    return intervals

def parse_date_price_data(data):
    """Parse data in format: MM/DD/YYYY value"""
    dates = []
    prices = []
    for line in data:
        try:
            parts = line.strip().split()
            if len(parts) >= 2:
                date_str = parts[0]
                price_str = parts[1]
                date = datetime.strptime(date_str.strip(), '%m/%d/%Y')
                price = float(price_str.strip())
                dates.append(date)
                prices.append(price)
        except (ValueError, IndexError):
            continue
    return dates, prices

def calculate_universe(prices):
    """Bước 1: Xác định tập nền U của chuỗi thời gian X(t)
    
    Cho tập nền U = [u_min, u_max] = [D_min - D1, D_max + D2], trong đó:
    - D_min, D_max là giá trị nhỏ nhất và lớn nhất của chuỗi dữ liệu
    - D1, D2 là hai số dương được chọn sao cho tập nền U chứa tất cả các giá trị dữ liệu quá khứ
    
    Trong trường hợp này:
    - D_min = 13055
    - D_max = 19337
    - D1 = 55
    - D2 = 663
    - U = [13000, 20000]
    """
    if not prices:
        return {
            'min': 0,
            'max': 0,
            'extended_min': 0,
            'extended_max': 0,
            'D1': 55,
            'D2': 663
        }
        
    dmin = min(prices)
    dmax = max(prices)
    d1 = 55  # Giá trị D1 cố định
    d2 = 663  # Giá trị D2 cố định
    u_min = dmin - d1
    u_max = dmax + d2
    
    return {
        'min': dmin,
        'max': dmax,
        'extended_min': u_min,
        'extended_max': u_max,
        'D1': d1,
        'D2': d2
    }

def create_fuzzy_sets(intervals):
    """Bước 3: Xác định các tập mờ cho các quan sát trong chuỗi thời gian
    
    Mỗi một khoảng được xác định trong Bước 2 biểu diễn một giá trị ngôn ngữ của biến ngôn ngữ tuyển sinh "enrolments".
    Dựa vào k khoảng chia, ta xác định được k giá trị ngôn ngữ. Mỗi nhãn ngôn ngữ là một tập mờ A_i được xác định theo công thức:
    
    A_i = a_i1/u_1 + a_i2/u_2 + ... + a_ij/u_j + ... + a_ik/u_k
    
    Trong đó:
    - a_ij ∈ [0,1] là cấp độ của uj vào tập mờ A_i
    - u_j là khoảng thứ j của tập nền
    - a_ij được xác định theo công thức:
        a_ij = 1     nếu j = i
        a_ij = 0.5   nếu j = i-1 hoặc j = i+1
        a_ij = 0     các trường hợp còn lại
    
    Args:
        intervals: List các tuple (start, end) đại diện cho các khoảng
        
    Returns:
        List các tập mờ, mỗi tập mờ là một dict chứa:
        - interval: Khoảng giá trị
        - membership: List các giá trị độ thuộc
        - label: Nhãn của tập mờ (A1, A2, ...)
    """
    if not intervals:
        return []
        
    fuzzy_sets = []
    # Sắp xếp các khoảng theo giá trị bắt đầu
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    for i, (start, end) in enumerate(sorted_intervals):
        if start is None or end is None:
            continue
            
        # Khởi tạo list độ thuộc với giá trị 0
        membership = [0] * len(intervals)
        
        # Gán giá trị độ thuộc theo công thức (2.7)
        membership[i] = 1.0  # a_ij = 1 khi j = i
        if i > 0:
            membership[i-1] = 0.5  # a_ij = 0.5 khi j = i-1
        if i < len(intervals)-1:
            membership[i+1] = 0.5  # a_ij = 0.5 khi j = i+1
            
        fuzzy_sets.append({
            'interval': [start, end],
            'membership': membership,
            'label': f'A{i+1}'  # Đảm bảo nhãn là A1, A2, A3,...
        })
    
    # Sắp xếp lại fuzzy_sets theo khoảng
    fuzzy_sets.sort(key=lambda x: x['interval'][0])
    return fuzzy_sets

def fuzzify_data(dates, prices, fuzzy_sets):
    """Bước 4: Mờ hóa dữ liệu chuỗi thời gian
    
    Để thực hiện mờ hóa chuỗi dữ liệu, trước tiên cần gán các giá trị ngôn ngữ tương ứng với từng tập mờ 
    cho mỗi khoảng chia cụ thể. Nếu giá trị lịch sử của biến chuỗi thời gian tại thời điểm t nằm trong 
    khoảng u_i (tức là Y(t) ∈ u_i) và mức độ thuộc cao nhất của tập mờ A_i xảy ra tại khoảng này, thì 
    dữ liệu của biến chuỗi thời gian sẽ được mờ hóa dưới dạng tập mờ A_i.
    
    Args:
        dates: List các ngày tháng
        prices: List các giá trị tương ứng với ngày tháng
        fuzzy_sets: List các tập mờ đã được xác định
        
    Returns:
        List các dict chứa thông tin mờ hóa:
        - date: Ngày tháng
        - original: Giá trị gốc
        - fuzzy_set: Nhãn tập mờ được gán
        - membership: List các giá trị độ thuộc
    """
    if not dates or not prices or not fuzzy_sets:
        return []
        
    fuzzified_data = []
    for date, price in zip(dates, prices):
        if price is None:
            continue
            
        max_membership = 0
        selected_set = None
        
        # Tìm tập mờ phù hợp nhất dựa trên khoảng và độ thuộc
        for fuzzy_set in fuzzy_sets:
            if not fuzzy_set.get('interval') or len(fuzzy_set['interval']) != 2:
                continue
                
            start, end = fuzzy_set['interval']
            # Kiểm tra nếu giá trị nằm trong khoảng
            if start <= price <= end:
                membership = fuzzy_set.get('membership', [])
                # Tìm độ thuộc cao nhất trong tập mờ
                current_max = max(membership)
                if current_max > max_membership:
                    max_membership = current_max
                    selected_set = fuzzy_set
        
        # Nếu không tìm thấy tập mờ phù hợp, tìm tập mờ gần nhất
        if not selected_set:
            min_distance = float('inf')
            for fuzzy_set in fuzzy_sets:
                if not fuzzy_set.get('interval') or len(fuzzy_set['interval']) != 2:
                    continue
                    
                mid_point = (fuzzy_set['interval'][0] + fuzzy_set['interval'][1]) / 2
                distance = abs(price - mid_point)
                if distance < min_distance:
                    min_distance = distance
                    selected_set = fuzzy_set
        
        if selected_set and selected_set.get('label'):
            fuzzified_data.append({
                'date': date.strftime('%m/%d/%Y'),
                'original': price,
                'fuzzy_set': selected_set['label'],
                'membership': selected_set.get('membership', [])
            })
    
    return fuzzified_data

def create_fuzzy_relations(fuzzified_data, order=1):
    """Bước 5: Xác định tất cả các quan hệ mờ bậc p (p ≥ 1)
    
    Từ các khái niệm QHM bậc 1 và bậc cao, ta cần tìm ra các quan hệ có dạng:
    F(t-p), F(t-p+1), ..., F(t-1) → F(t)
    
    Trong đó:
    - F(t-p), F(t-p+1), ..., F(t-1) là trạng thái hiện tại
    - F(t) là trạng thái tương lai của quan hệ mờ
    
    Quan hệ này được thay thế bởi quan hệ mờ với các nhãn ngôn ngữ:
    A_ip, A_i(p-1), ..., A_i2, A_i1 → A_k
    
    Ví dụ:
    - QHM bậc 1: A_2 → A_3 (từ F(1993) → F(1994))
    - QHM bậc 3: A_1, A_1, A_2 → A_3 (từ F(1971), F(1972), F(1973) → F(1974))
    
    Args:
        fuzzified_data: List các dict chứa thông tin mờ hóa
        order: Bậc của quan hệ mờ (mặc định là 1)
        
    Returns:
        Dict chứa các quan hệ mờ:
        - order1: List các quan hệ mờ bậc 1
        - order3: List các quan hệ mờ bậc 3
    """
    if not fuzzified_data or len(fuzzified_data) < 2:
        return {'order1': [], 'order3': []}
        
    relations = []
    
    # Quan hệ bậc 1
    for i in range(len(fuzzified_data) - 1):
        current_state = fuzzified_data[i].get('fuzzy_set')
        next_state = fuzzified_data[i+1].get('fuzzy_set')
        
        if current_state and next_state:  # Chỉ thêm quan hệ nếu cả hai trạng thái đều tồn tại
            relations.append({
                'current': current_state,  # Lưu dưới dạng string
                'next': next_state,
                'date': fuzzified_data[i+1].get('date')
            })
    
    # Quan hệ bậc 3
    order3_relations = []
    if len(fuzzified_data) >= 4:  # Cần ít nhất 4 điểm dữ liệu để tạo quan hệ bậc 3
        for i in range(len(fuzzified_data) - 3):
            current_states = [
                fuzzified_data[i].get('fuzzy_set'),
                fuzzified_data[i+1].get('fuzzy_set'),
                fuzzified_data[i+2].get('fuzzy_set')
            ]
            next_state = fuzzified_data[i+3].get('fuzzy_set')
            
            if all(current_states) and next_state:  # Chỉ thêm quan hệ nếu tất cả trạng thái đều tồn tại
                order3_relations.append({
                    'current': current_states,  # Lưu dưới dạng list
                    'next': next_state,
                    'date': fuzzified_data[i+3].get('date')
                })
    
    return {
        'order1': relations,
        'order3': order3_relations
    }

def create_fuzzy_groups(relations):
    """Bước 6: Thiết lập NQHM-PTTG bậc p
    
    Sử dụng khái niệm nhóm quan hệ mờ để thiết lập nhóm quan hệ mờ phụ thuộc thời gian.
    
    Ví dụ:
    - t = 1972: Nhóm 1: A_1 → A_1
    - t = 1973: Nhóm 2: A_1 → A_1, A_2
    - t = 1974: Nhóm 3: A_1 → A_1, A_2, A_3
    - t = 1975: Nhóm 4: A_3 → A_4
    
    Args:
        relations: Dict chứa các quan hệ mờ:
            - order1: List các quan hệ mờ bậc 1
            - order3: List các quan hệ mờ bậc 3
            
    Returns:
        List các dict chứa thông tin nhóm quan hệ mờ:
        - date: Ngày tháng
        - group_count: Số thứ tự nhóm
        - current_fuzzy_set: Tập mờ hiện tại
        - order1_relations: Các quan hệ mờ bậc 1
        - order3_relations: Các quan hệ mờ bậc 3
    """
    # Tạo bảng theo ngày
    date_groups = {}
    
    # Xử lý quan hệ bậc 1
    for relation in relations['order1']:
        if not relation.get('current') or not relation.get('next') or not relation.get('date'):
            continue
            
        date = relation['date']
        current = relation['current']  # Đã là string
        
        if date not in date_groups:
            date_groups[date] = {
                'date': date,
                'group_count': 0,
                'current_fuzzy_set': current,
                'order1_relations': {},  # Dict để gom nhóm theo vế trái
                'order3_relations': set()
            }
        
        # Gom nhóm các quan hệ có cùng vế trái
        if current not in date_groups[date]['order1_relations']:
            date_groups[date]['order1_relations'][current] = set()
        date_groups[date]['order1_relations'][current].add(relation['next'])
    
    # Xử lý quan hệ bậc 3
    for relation in relations['order3']:
        if not relation.get('current') or not relation.get('next') or not relation.get('date'):
            continue
            
        date = relation['date']
        current = relation['current'][-1] if isinstance(relation['current'], list) else relation['current']
        
        if date not in date_groups:
            date_groups[date] = {
                'date': date,
                'group_count': 0,
                'current_fuzzy_set': current,
                'order1_relations': {},
                'order3_relations': set()
            }
        
        # Format quan hệ mờ bậc 3
        current_states = ', '.join(relation['current']) if isinstance(relation['current'], list) else relation['current']
        relation_str = f"{current_states} → {relation['next']}"
        date_groups[date]['order3_relations'].add(relation_str)
    
    # Sắp xếp theo ngày và đánh số nhóm
    sorted_dates = sorted(date_groups.keys())
    for i, date in enumerate(sorted_dates, 1):
        date_groups[date]['group_count'] = i
    
    # Chuyển đổi thành danh sách và format kết quả
    result = []
    for date in sorted_dates:
        group = date_groups[date]
        
        # Format quan hệ bậc 1: gom nhóm theo vế trái
        order1_relations = []
        for left_side, right_sides in sorted(group['order1_relations'].items()):
            right_sides_str = ', '.join(sorted(right_sides))
            order1_relations.append(f"{left_side} → {right_sides_str}")
        
        # Format quan hệ bậc 3
        order3_relations = sorted(list(group['order3_relations']))
        
        # Format kết quả
        result.append({
            'date': group['date'],
            'group_count': group['group_count'],
            'current_fuzzy_set': group['current_fuzzy_set'],
            'order1_relations': ' | '.join(order1_relations) if order1_relations else '-',
            'order3_relations': ' | '.join(order3_relations) if order3_relations else '-'
        })
    
    return result

def defuzzify(predicted_set, fuzzy_sets, q=10, w_h=15):
    """Bước 7: Giải mờ và tính toán giá trị đầu ra dự báo
    
    Thực hiện giải mờ dữ liệu và tính toán giá trị dự báo cho các nhóm quan hệ mờ bậc một và bậc cao
    theo 3 quy tắc:
    
    Quy tắc 1: Trường hợp nhóm quan hệ mờ bậc 1 (p = 1)
    Giá trị dự báo được tính theo công thức:
    Giá trị_DB = (1*M_i1 + 2*M_i2 + ... + m*M_im) / (1 + 2 + ... + m)
    
    Trong đó:
    - M_i1, M_i2, ..., M_ik là điểm giữa của các khoảng u_i1, u_i2, ..., u_ik
    - k (1 ≤ k ≤ m) là các trọng số theo vị trí xuất hiện
    
    Quy tắc 2: Trường hợp nhóm quan hệ mờ bậc cao (p ≥ 2)
    Giá trị dự báo được tính theo công thức:
    Giá trị_DB = 1/m * ∑(i=1 to m) subm_ik
    
    Trong đó:
    - m là tổng số tập mờ bên vế phải
    - subm_ik là điểm giữa của khoảng con thứ k
    
    Quy tắc 3: Trường hợp nhóm quan hệ mờ có vế phải chưa xác định tập mờ
    Giá trị dự báo được tính theo công thức:
    Giá trị_DB(#) = (M_t1*w_h + M_t2 + ... + M_ti + ... + M_tp) / (w_h + (p-1))
    
    Trong đó:
    - w_h là trọng số cao nhất (mặc định = 15)
    - M_t1, M_ti là điểm giữa của các khoảng
    
    Args:
        predicted_set: Tập mờ dự đoán
        fuzzy_sets: List các tập mờ
        q: Số khoảng con (mặc định = 10)
        w_h: Trọng số cao nhất (mặc định = 15)
        
    Returns:
        float: Giá trị dự báo sau khi giải mờ
    """
    if not predicted_set or not fuzzy_sets:
        return 0.0
        
    # Tìm tập mờ phù hợp
    target_set = None
    for fuzzy_set in fuzzy_sets:
        if fuzzy_set['label'] == predicted_set:
            target_set = fuzzy_set
            break
    
    if not target_set:
        return 0.0
    
    # Tính điểm giữa của khoảng
    start, end = target_set['interval']
    mid_point = (start + end) / 2
    
    # Nếu là dấu # (quy tắc 3)
    if predicted_set == '#':
        # Lấy tất cả các tập mờ từ quan hệ bậc 3
        all_sets = []
        for fuzzy_set in fuzzy_sets:
            if fuzzy_set['label'] != '#':
                all_sets.append(fuzzy_set)
        
        if all_sets:
            # Tính tổng trọng số và tổng tích
            total_weight = w_h + (len(all_sets) - 1)
            weighted_sum = 0
            
            # Lấy điểm giữa của các tập mờ
            mid_points = []
            for set_label in all_sets:
                for fuzzy_set in fuzzy_sets:
                    if fuzzy_set['label'] == set_label:
                        mid_point = (fuzzy_set['interval'][0] + fuzzy_set['interval'][1]) / 2
                        mid_points.append(mid_point)
                        break
            
            if mid_points:
                # Tính giá trị dự đoán theo công thức (2.10)
                weighted_sum = w_h * mid_points[0]  # Tập mờ xuất hiện gần nhất
                for mid_point in mid_points[1:]:  # Các tập mờ còn lại
                    weighted_sum += mid_point
                
                if total_weight > 0:
                    return weighted_sum / total_weight
    
    # Trường hợp mặc định: lấy điểm giữa của khoảng
    return mid_point

def predict_next_days(last_date, last_price, fuzzy_groups, fuzzy_sets, num_days=10):
    """Dự đoán giá cho các ngày tiếp theo"""
    return []  # Bỏ phần dự đoán 10 ngày

def kmeans_clustering(prices, k=7):
    """Bước 2: Áp dụng thuật toán K-means để chia tập EUA thành k khoảng có độ dài khác nhau
    
    Args:
        prices: Danh sách các giá trị cần chia khoảng
        k: Số khoảng cần chia (mặc định là 7)
        
    Returns:
        List các tuple (start, end) đại diện cho các khoảng
    """
    # Chuẩn bị dữ liệu
    X = np.array(prices).reshape(-1, 1)
    
    # Áp dụng KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    # Lấy tâm cụm và sắp xếp theo thứ tự tăng dần
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    # Tính toán các khoảng dựa trên tâm cụm
    intervals = []
    for i in range(len(centers)-1):
        mid_point = (centers[i] + centers[i+1]) / 2
        intervals.append((centers[i], mid_point))
    
    # Thêm khoảng đầu và cuối
    first_interval = (centers[0] - (centers[1] - centers[0]), centers[0])
    last_interval = (centers[-1], centers[-1] + (centers[-1] - centers[-2]))
    intervals.insert(0, first_interval)
    intervals.append(last_interval)
    
    return intervals

def equal_width_clustering(prices, k=7):
    """Bước 2: Áp dụng phương pháp chiều rộng bằng nhau để chia tập EUA thành k khoảng
    
    Args:
        prices: Danh sách các giá trị cần chia khoảng
        k: Số khoảng cần chia (mặc định là 7)
        
    Returns:
        List các tuple (start, end) đại diện cho các khoảng
    """
    min_price = min(prices)
    max_price = max(prices)
    width = (max_price - min_price) / k
    
    intervals = []
    for i in range(k):
        start = min_price + i * width
        end = start + width
        intervals.append((start, end))
    
    return intervals

def equal_frequency_clustering(prices, k=7):
    """Bước 2: Áp dụng phương pháp tần suất bằng nhau để chia tập EUA thành k khoảng
    
    Args:
        prices: Danh sách các giá trị cần chia khoảng
        k: Số khoảng cần chia (mặc định là 7)
        
    Returns:
        List các tuple (start, end) đại diện cho các khoảng
    """
    sorted_prices = sorted(prices)
    n = len(sorted_prices)
    points_per_interval = n // k
    
    intervals = []
    for i in range(k):
        start_idx = i * points_per_interval
        end_idx = start_idx + points_per_interval if i < k-1 else n
        start = sorted_prices[start_idx]
        end = sorted_prices[end_idx-1]
        intervals.append((start, end))
    
    return intervals

def format_fuzzy_groups_table(fuzzy_groups):
    """Format fuzzy groups into a readable table structure"""
    return fuzzy_groups  # Đã được format trong create_fuzzy_groups

def fuzzy_time_series_prediction(data, clustering_method='kmeans', num_intervals=7):
    """Thực hiện dự đoán chuỗi thời gian mờ"""
    if not data:
        return {
            'universe': calculate_universe([]),
            'intervals': [],
            'fuzzy_sets': [],
            'fuzzified_data': [],
            'fuzzy_relations': {'order1': [], 'order3': []},
            'fuzzy_groups': [],
            'fuzzy_groups_table': [],
            'predictions': []
        }
    
    # Bước 1: Parse dữ liệu ngày và giá
    dates, prices = parse_date_price_data(data)
    if not dates or not prices:
        return {
            'universe': calculate_universe([]),
            'intervals': [],
            'fuzzy_sets': [],
            'fuzzified_data': [],
            'fuzzy_relations': {'order1': [], 'order3': []},
            'fuzzy_groups': [],
            'fuzzy_groups_table': [],
            'predictions': []
        }
    
    # Bước 2: Xác định miền vũ trụ
    universe = calculate_universe(prices)
    
    # Bước 3: Chia khoảng dựa trên phương pháp được chọn
    if clustering_method == 'kmeans':
        intervals = kmeans_clustering(prices, k=num_intervals)
    elif clustering_method == 'kmeans_pso':
        intervals = hybrid_kmeans_pso(prices, k=num_intervals)
    elif clustering_method == 'equal_width':
        intervals = equal_width_clustering(prices, k=num_intervals)
    elif clustering_method == 'equal_frequency':
        intervals = equal_frequency_clustering(prices, k=num_intervals)
    else:
        intervals = kmeans_clustering(prices, k=num_intervals)  # Mặc định sử dụng K-means
    
    if not intervals:
        return {
            'universe': universe,
            'intervals': [],
            'fuzzy_sets': [],
            'fuzzified_data': [],
            'fuzzy_relations': {'order1': [], 'order3': []},
            'fuzzy_groups': [],
            'fuzzy_groups_table': [],
            'predictions': []
        }
    
    # Bước 4: Xác định tập mờ
    fuzzy_sets = create_fuzzy_sets(intervals)
    if not fuzzy_sets:
        return {
            'universe': universe,
            'intervals': intervals,
            'fuzzy_sets': [],
            'fuzzified_data': [],
            'fuzzy_relations': {'order1': [], 'order3': []},
            'fuzzy_groups': [],
            'fuzzy_groups_table': [],
            'predictions': []
        }
    
    # Bước 5: Fuzzify dữ liệu
    fuzzified_data = fuzzify_data(dates, prices, fuzzy_sets)
    if not fuzzified_data:
        return {
            'universe': universe,
            'intervals': intervals,
            'fuzzy_sets': fuzzy_sets,
            'fuzzified_data': [],
            'fuzzy_relations': {'order1': [], 'order3': []},
            'fuzzy_groups': [],
            'fuzzy_groups_table': [],
            'predictions': []
        }
    
    # Bước 6: Xác định quan hệ mờ
    fuzzy_relations = create_fuzzy_relations(fuzzified_data)
    
    # Bước 7: Tạo nhóm quan hệ mờ
    fuzzy_groups = create_fuzzy_groups(fuzzy_relations)
    
    # Dự đoán cho dữ liệu hiện có
    predictions = []
    recent_states = []  # Lưu lại 3 trạng thái gần nhất
    
    for i in range(len(prices)-1):
        current_date = dates[i]
        current_price = prices[i]
        actual_next = prices[i+1]
        
        # Tìm tập mờ hiện tại
        current_fuzzy_set = None
        for fuzzy_set in fuzzy_sets:
            if not fuzzy_set.get('interval') or len(fuzzy_set['interval']) != 2:
                continue
                
            start, end = fuzzy_set['interval']
            if start <= current_price <= end:
                current_fuzzy_set = fuzzy_set.get('label')
                break
        
        # Nếu không tìm thấy tập mờ phù hợp, sử dụng tập mờ gần nhất
        if not current_fuzzy_set:
            min_distance = float('inf')
            for fuzzy_set in fuzzy_sets:
                if not fuzzy_set.get('interval') or len(fuzzy_set['interval']) != 2:
                    continue
                    
                mid_point = (fuzzy_set['interval'][0] + fuzzy_set['interval'][1]) / 2
                distance = abs(current_price - mid_point)
                if distance < min_distance:
                    min_distance = distance
                    current_fuzzy_set = fuzzy_set.get('label')
        
        if not current_fuzzy_set:
            continue
        
        # Cập nhật danh sách trạng thái gần nhất
        recent_states.append(current_fuzzy_set)
        if len(recent_states) > 3:
            recent_states.pop(0)
        
        # Dự đoán bậc 1
        order1_predicted = None
        order1_rule = None
        if current_fuzzy_set:
            for group in fuzzy_groups:
                if not group.get('order1_relations'):
                    continue
                    
                relations = group['order1_relations'].split(', ')
                for relation in relations:
                    if not relation.startswith(current_fuzzy_set):
                        continue
                        
                    parts = relation.split(' → ')
                    if len(parts) == 2:
                        predicted_set = parts[1]
                        order1_predicted = defuzzify(predicted_set, fuzzy_sets)
                        order1_rule = 'Quy tắc 1'
                        break
                if order1_predicted:
                    break
        
        # Dự đoán bậc 3
        order3_predicted = None
        order3_rule = None
        if len(recent_states) == 3:
            current_key = ','.join(recent_states)
            for group in fuzzy_groups:
                if not group.get('order3_relations'):
                    continue
                    
                relations = group['order3_relations'].split(', ')
                for relation in relations:
                    if not relation.startswith(current_key):
                        continue
                        
                    parts = relation.split(' → ')
                    if len(parts) == 2:
                        predicted_set = parts[1]
                        order3_predicted = defuzzify(predicted_set, fuzzy_sets)
                        order3_rule = 'Quy tắc 2'
                        break
                if order3_predicted:
                    break
        
        # Nếu không có dự đoán nào, sử dụng quy tắc 3
        if not order1_predicted and not order3_predicted:
            order1_predicted = defuzzify('#', fuzzy_sets)
            order3_predicted = defuzzify('#', fuzzy_sets)
            order1_rule = 'Quy tắc 3'
            order3_rule = 'Quy tắc 3'
        
        # Nếu vẫn không có dự đoán, sử dụng giá trị hiện tại
        if not order1_predicted:
            order1_predicted = current_price
            order1_rule = 'Giá trị hiện tại'
        if not order3_predicted:
            order3_predicted = current_price
            order3_rule = 'Giá trị hiện tại'
        
        predictions.append({
            'date': dates[i+1].strftime('%m/%d/%Y'),
            'actual': float(actual_next) if actual_next is not None else 0.0,  # Đảm bảo giá trị là số
            'fuzzy_set': current_fuzzy_set,
            'order1_predicted': float(order1_predicted),  # Đảm bảo giá trị là số
            'order3_predicted': float(order3_predicted),  # Đảm bảo giá trị là số
            'order1_rule': order1_rule,
            'order3_rule': order3_rule
        })
    
    # Thêm dự đoán cuối cùng sử dụng quy tắc 3
    if prices and dates and fuzzy_sets and fuzzy_groups:
        last_date = dates[-1]
        last_price = prices[-1]
        
        # Tính toán khoảng cách trung bình giữa các mốc thời gian
        time_intervals = []
        for i in range(len(dates)-1):
            interval = (dates[i+1] - dates[i]).days
            time_intervals.append(interval)
        
        # Tính khoảng cách trung bình
        avg_interval = sum(time_intervals) / len(time_intervals) if time_intervals else 1
        
        # Tìm tập mờ hiện tại
        current_fuzzy_set = None
        for fuzzy_set in fuzzy_sets:
            if not fuzzy_set.get('interval') or len(fuzzy_set['interval']) != 2:
                continue
                
            start, end = fuzzy_set['interval']
            if start <= last_price <= end:
                current_fuzzy_set = fuzzy_set.get('label')
                break
        
        # Nếu không tìm thấy tập mờ phù hợp, sử dụng tập mờ gần nhất
        if not current_fuzzy_set:
            min_distance = float('inf')
            for fuzzy_set in fuzzy_sets:
                if not fuzzy_set.get('interval') or len(fuzzy_set['interval']) != 2:
                    continue
                    
                mid_point = (fuzzy_set['interval'][0] + fuzzy_set['interval'][1]) / 2
                distance = abs(last_price - mid_point)
                if distance < min_distance:
                    min_distance = distance
                    current_fuzzy_set = fuzzy_set.get('label')
        
        if current_fuzzy_set:
            # Tính toán dự đoán bậc 1 và bậc 3
            order1_predicted = None
            order3_predicted = None
            order1_rule = 'Quy tắc 3'
            order3_rule = 'Quy tắc 3'
            
            # Sử dụng quy tắc 3 để tính giá trị dự đoán
            if current_fuzzy_set:
                # Lấy tất cả các tập mờ từ quan hệ bậc 3
                all_sets = []
                for group in fuzzy_groups:
                    if group.get('order3_relations'):
                        for relation in group['order3_relations'].split(', '):
                            parts = relation.split(' → ')
                            if len(parts) == 2:
                                all_sets.extend(parts[0].split(', '))
                
                if all_sets:
                    # Tính tổng trọng số và tổng tích
                    w_h = 15  # Trọng số cho tập mờ xuất hiện gần nhất
                    total_weight = w_h + (len(all_sets) - 1)
                    weighted_sum = 0
                    
                    # Lấy điểm giữa của các tập mờ
                    mid_points = []
                    for set_label in all_sets:
                        for fuzzy_set in fuzzy_sets:
                            if fuzzy_set['label'] == set_label:
                                mid_point = (fuzzy_set['interval'][0] + fuzzy_set['interval'][1]) / 2
                                mid_points.append(mid_point)
                                break
                    
                    if mid_points:
                        # Tính giá trị dự đoán theo công thức
                        weighted_sum = w_h * mid_points[0]  # Tập mờ xuất hiện gần nhất
                        for mid_point in mid_points[1:]:  # Các tập mờ còn lại
                            weighted_sum += mid_point
                        
                        if total_weight > 0:
                            predicted_value = weighted_sum / total_weight
                            order1_predicted = predicted_value
                            order3_predicted = predicted_value
            
            # Nếu không có dự đoán, sử dụng giá trị hiện tại
            if not order1_predicted:
                order1_predicted = last_price
                order1_rule = 'Giá trị hiện tại'
            if not order3_predicted:
                order3_predicted = last_price
                order3_rule = 'Giá trị hiện tại'
            
            # Thêm ngày tiếp theo dựa trên khoảng cách trung bình
            next_date = last_date + timedelta(days=round(avg_interval))
            
            # Thêm dự đoán cuối cùng với dấu #
            predictions.append({
                'date': next_date.strftime('%m/%d/%Y'),
                'actual': 0.0,  # Giá trị mặc định cho dự đoán tương lai
                'fuzzy_set': '#',  # Sử dụng dấu # cho dự đoán cuối cùng
                'order1_predicted': float(order1_predicted),  # Đảm bảo giá trị là số
                'order3_predicted': float(order3_predicted),  # Đảm bảo giá trị là số
                'order1_rule': order1_rule,
                'order3_rule': order3_rule
            })
    
    return {
        'universe': universe,
        'intervals': intervals,
        'fuzzy_sets': fuzzy_sets,
        'fuzzified_data': fuzzified_data,
        'fuzzy_relations': fuzzy_relations,
        'fuzzy_groups': fuzzy_groups,
        'fuzzy_groups_table': fuzzy_groups,
        'predictions': predictions
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('data', [])
    clustering_method = request.json.get('clustering_method', 'kmeans')
    num_intervals = request.json.get('num_intervals', 7)
    result = fuzzy_time_series_prediction(data, clustering_method=clustering_method, num_intervals=num_intervals)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
