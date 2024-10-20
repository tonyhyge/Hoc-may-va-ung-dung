# 1.Công Nghệ Sử dụng:

---

+ Framawork: pandas, numpy, flask, io, matplotlib, 


# 2. Thuật toán:
---

### Decision Tree Model
Decision Tree là một thuật toán học máy sử dụng cấu trúc cây để đưa ra quyết định. Nó phân chia dữ liệu thành các nhánh dựa trên các thuộc tính của dữ liệu. Mỗi nút trong cây đại diện cho một thuộc tính, mỗi nhánh đại diện cho một quyết định hoặc kết quả của thuộc tính đó. Cây quyết định có thể sử dụng cho cả bài toán phân loại (classification) và hồi quy (regression).

Ưu điểm:

Dễ hiểu và dễ giải thích.
Không yêu cầu dữ liệu chuẩn hóa.
Có thể xử lý cả dữ liệu phân loại và hồi quy.
Nhược điểm:

Dễ bị overfitting nếu cây quá sâu.
Nhạy cảm với sự thay đổi trong dữ liệu.

### Random Forest
Random Forest là một tập hợp  của nhiều cây quyết định. Nó tạo ra nhiều cây quyết định trên các mẫu dữ liệu khác nhau và kết hợp các dự đoán của chúng để cải thiện độ chính xác và giảm overfitting. Random Forest thường sử dụng phương pháp bỏ phiếu (voting) để đưa ra kết quả.

Ưu điểm:

Có khả năng tổng quát tốt hơn so với một cây quyết định đơn lẻ.
Giảm thiểu overfitting.
Khả năng xử lý dữ liệu lớn và nhiều thuộc tính.
Nhược điểm:

Thời gian huấn luyện và dự đoán lâu hơn so với một cây quyết định đơn lẻ.
Khó giải thích hơn vì nó bao gồm nhiều cây.


# 3. Hiển thị kết quả lên website

---

<img width="1195" alt="Ảnh màn hình 2024-10-20 lúc 19 32 25" src="https://github.com/user-attachments/assets/6e7505c3-b190-435c-837d-75fcbb114c70">


# 4. So sánh 2 thuật toán 

|Tiêu chí	|Decision Tree|	Random Forest|
|---------|-------------|--------------|
|Cấu trúc	|Một cây 	|Tập hợp của nhiều cây|
|Độ chính xác	|Có thể thấp do overfitting|	 cao hơn|
|Thời gian huấn luyện|	Nhanh	|Chậm hơn
|Xử lý thiếu dữ liệu|	nhạy cảm 	|Tốt hơn do có nhiều cây|
