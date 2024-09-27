1. Công nghệ sử dụng:
• Framework:numpy, pandas, matplotlib, io
2. Thuật toán:
Hồi quy tuyến tính (linear regression) bằng cách sử dụng phương trình chuẩn để dự đoán mối quan hệ giữa chiều cao và cân nặng
• Giải thích (giải thích theo suy nghĩ của mình hiểu về thuật toán này hoạt động. Yêu cầu này ai làm được thì tốt).
   X là mảng numpy chưa danh sách chiều cao định dạng là 1 vector cột
   y là mảng numnpy dạng vector cột chưa giá trị cân nặng
   X = np.insert(X, 0, 1, axis=1) thêm một cột toàn giá trị 1 vào X như cột đầu tiên thêm một hệ số hằng (intercept) vào mô hình
   theta = np.linalg.inv(X.T @ X) @ (X.T @ y) tính tham số tối ưu theta
  dự đoán thử:
  x1=150 x2 = 190
   y1 tính cân nặng x1 (tương tự y2)
4. Hiển thị kết quả lên website
• Chụp ảnh và dán kết quả tại đây.
<img width="1523" alt="Ảnh màn hình 2024-09-27 lúc 11 08 45" src="https://github.com/user-attachments/assets/77df061e-e63b-423d-9bbf-9d7785ecce38">
5. Đối với các bài toán có sự so sánh giữa 2 thuật toán thì sẽ dán kết quả dưới đây.
