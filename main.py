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
    """Kết hợp K-means và PSO để tối ưu hóa phân cụm"""
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
    """Bước 1: Xác định tập nền U"""
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
        'extended_max': u_max
    }

def create_fuzzy_sets(intervals):
    """Bước 3: Xác định các tập mờ"""
    fuzzy_sets = []
    for i, (start, end) in enumerate(intervals):
        membership = [0] * len(intervals)
        if i > 0:
            membership[i-1] = 0.5
        membership[i] = 1
        if i < len(intervals)-1:
            membership[i+1] = 0.5
            
        fuzzy_sets.append({
            'interval': [start, end],
            'membership': membership,
            'label': f'A{i+1}'
        })
    return fuzzy_sets

def fuzzify_data(dates, prices, fuzzy_sets):
    """Bước 4: Mờ hóa dữ liệu"""
    fuzzified_data = []
    for date, price in zip(dates, prices):
        max_membership = 0
        selected_set = None
        
        for fuzzy_set in fuzzy_sets:
            start, end = fuzzy_set['interval']
            if start <= price <= end:
                membership = fuzzy_set['membership']
                if max(membership) > max_membership:
                    max_membership = max(membership)
                    selected_set = fuzzy_set
        
        if selected_set:
            fuzzified_data.append({
                'date': date.strftime('%m/%d/%Y'),
                'original': price,
                'fuzzy_set': selected_set['label'],
                'membership': selected_set['membership']
            })
    
    return fuzzified_data

def create_fuzzy_relations(fuzzified_data, order=1):
    """Bước 5: Xác định quan hệ mờ bậc p"""
    relations = []
    for i in range(len(fuzzified_data) - order):
        current_state = []
        for j in range(order):
            current_state.append(fuzzified_data[i+j]['fuzzy_set'])
        next_state = fuzzified_data[i+order]['fuzzy_set']
        
        relations.append({
            'current': current_state,
            'next': next_state
        })
    
    return relations

def create_fuzzy_groups(relations):
    """Bước 6: Thiết lập nhóm quan hệ mờ"""
    groups = {}
    for relation in relations:
        current_key = ','.join(relation['current'])
        if current_key not in groups:
            groups[current_key] = set()
        groups[current_key].add(relation['next'])
    
    return [{'current': k.split(','), 'next': list(v)} for k, v in groups.items()]

def defuzzify(predicted_set, fuzzy_sets):
    """Bước 7: Giải mờ và tính toán giá trị dự báo"""
    for fuzzy_set in fuzzy_sets:
        if fuzzy_set['label'] == predicted_set:
            return (fuzzy_set['interval'][0] + fuzzy_set['interval'][1]) / 2
    return None

def predict_next_days(last_date, last_price, fuzzy_groups, fuzzy_sets, num_days=10):
    """Dự đoán giá cho các ngày tiếp theo"""
    predictions = []
    current_price = last_price
    current_date = last_date
    
    for _ in range(num_days):
        # Tìm tập mờ hiện tại
        current_fuzzy_set = None
        for fuzzy_set in fuzzy_sets:
            start, end = fuzzy_set['interval']
            if start <= current_price <= end:
                current_fuzzy_set = fuzzy_set['label']
                break
        
        # Tìm tập mờ tiếp theo dựa trên nhóm quan hệ
        predicted_set = None
        for group in fuzzy_groups:
            if group['current'][-1] == current_fuzzy_set:
                predicted_set = group['next'][0]  # Lấy phần tử đầu tiên của tập next
                break
        
        # Defuzzify để có giá trị dự đoán
        if predicted_set:
            predicted_value = defuzzify(predicted_set, fuzzy_sets)
        else:
            predicted_value = current_price
        
        # Cập nhật ngày và giá cho lần dự đoán tiếp theo
        current_date = current_date + timedelta(days=1)
        current_price = predicted_value
        
        predictions.append({
            'date': current_date.strftime('%m/%d/%Y'),
            'predicted': predicted_value
        })
    
    return predictions

def kmeans_clustering(prices, k=7):
    """Áp dụng K-means để chia tập thành k khoảng"""
    # Chuẩn bị dữ liệu
    X = np.array(prices).reshape(-1, 1)
    
    # Áp dụng KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    # Lấy tâm cụm
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
    """Chia khoảng bằng phương pháp chiều rộng bằng nhau"""
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
    """Chia khoảng bằng phương pháp tần suất bằng nhau"""
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

def fuzzy_time_series_prediction(data, model_type='kmeans', clustering_method='kmeans', num_intervals=7):
    # Bước 1: Parse dữ liệu ngày và giá
    dates, prices = parse_date_price_data(data)
    
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
    
    # Bước 4: Xác định tập mờ
    fuzzy_sets = create_fuzzy_sets(intervals)
    
    # Bước 5: Fuzzify dữ liệu
    fuzzified_data = fuzzify_data(dates, prices, fuzzy_sets)
    
    # Bước 6: Xác định quan hệ mờ
    fuzzy_relations = create_fuzzy_relations(fuzzified_data)
    
    # Bước 7: Tạo nhóm quan hệ mờ
    fuzzy_groups = create_fuzzy_groups(fuzzy_relations)
    
    # Dự đoán cho dữ liệu hiện có
    predictions = []
    for i in range(len(prices)-1):
        current_date = dates[i]
        current_price = prices[i]
        actual_next = prices[i+1]
        
        # Tìm tập mờ hiện tại
        current_fuzzy_set = None
        for fuzzy_set in fuzzy_sets:
            start, end = fuzzy_set['interval']
            if start <= current_price <= end:
                current_fuzzy_set = fuzzy_set['label']
                break
        
        # Tìm tập mờ tiếp theo dựa trên nhóm quan hệ
        predicted_set = None
        for group in fuzzy_groups:
            if group['current'][-1] == current_fuzzy_set:
                predicted_set = group['next'][0]
                break
        
        # Defuzzify để có giá trị dự đoán
        if predicted_set:
            predicted_value = defuzzify(predicted_set, fuzzy_sets)
        else:
            predicted_value = current_price
            
        predictions.append({
            'date': dates[i+1].strftime('%m/%d/%Y'),
            'actual': actual_next,
            'predicted': predicted_value
        })
    
    # Dự đoán 10 ngày tiếp theo
    future_predictions = predict_next_days(
        dates[-1],
        prices[-1],
        fuzzy_groups,
        fuzzy_sets,
        num_days=10
    )
    
    return {
        'universe': universe,
        'intervals': intervals,
        'fuzzy_sets': fuzzy_sets,
        'fuzzified_data': fuzzified_data,
        'fuzzy_relations': fuzzy_relations,
        'fuzzy_groups': fuzzy_groups,
        'predictions': predictions,
        'future_predictions': future_predictions
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('data', [])
    model_type = request.json.get('model', 'kmeans')
    clustering_method = request.json.get('clustering_method', 'kmeans')
    num_intervals = request.json.get('num_intervals', 7)
    result = fuzzy_time_series_prediction(data, model_type=model_type, clustering_method=clustering_method, num_intervals=num_intervals)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
