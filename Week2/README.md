**1. Công nghệ sử dụng:**

• Framework:numpy, pandas, matplotlib, io, sklearn, Flask, 


**2. Thuật toán:**

• Thuật toán Naive Bayes 
  - Một thuật toán học máy có giám sát (Supervised learning). Dự đoán xác suất sau của bất kỳ sự kiện nào dựa trên các sự kiện đã xảy ra. Naive Bayes được sử dụng để thực hiện phân loại và giả định rằng tất cả các sự kiện đều độc lập. Naive Bayes  được sử dụng để tính xác suất có điều kiện của một sự kiện, với điều kiện là một sự kiện khác đã xảy ra.

• Bernoulli Naive Bayes
  - Mô hình này được sử dụng khi các đặc trưng đầu vào chỉ nhận giá trị nhị phân 0 hoặc 1 (phân bố Bernoulli). Phù hợp để xử lý phân loại tài liệu 
  - Công thức : 
  - <img width="380" alt="Ảnh màn hình 2024-09-30 lúc 08 11 51" src="https://github.com/user-attachments/assets/f70b5cda-0d7a-4988-853d-b155a9f0eac0">


    
• Multinomial Naive Bayes
Mô thường được sử dụng cho các bài toán phân loại có đặc trưng là số nguyên không âm, đặc biệt phổ biến trong phân loại văn bản và mô hình phân tích tần suất của các từ trong tài liệu.

  - Công thức : 
  - <img width="376" alt="Ảnh màn hình 2024-09-30 lúc 08 12 34" src="https://github.com/user-attachments/assets/f6df8084-3497-4abe-8fea-7fff71264619">

• GaussianNB Naive Bayes'  
  Được sử dụng khi các features là dữ liệu liên tục, các giá trị của đặc trưng có thể nằm trên một dải rộng thay vì chỉ là nhị phân hoặc số nguyên.
  - Công thức : 
  <img width="534" alt="Ảnh màn hình 2024-09-30 lúc 09 58 50" src="https://github.com/user-attachments/assets/ddeb8f15-97f8-48b2-892e-5ac9de5975e2">

**3. Hiển thị kết quả lên website**

  - <img width="1662" alt="Ảnh màn hình 2024-09-30 lúc 08 01 46" src="https://github.com/user-attachments/assets/24e060e8-05b2-4eaa-bf5c-8c9dc5bdb2a7">


  - <img width="1468" alt="Ảnh màn hình 2024-09-30 lúc 09 54 29" src="https://github.com/user-attachments/assets/cf330f7e-e994-47ab-b9db-b2431ab91545">

**4. Đối với các bài toán có sự so sánh giữa 2 thuật toán thì sẽ dán kết quả dưới đây.**

  - ** so sánh giữa hai mô hình Bernoulli Naive Bayes và Multinomial Naive Bayes **
  - <img width="361" alt="Ảnh màn hình 2024-09-30 lúc 10 14 41" src="https://github.com/user-attachments/assets/89c90b27-8a53-473d-84a9-3501c0bdd806">
  - Multinomial Naive Bayes có độ chính xác (accuracy), recall, và F1-score cao hơn so với Bernoulli Naive Bayes. Với tập dữ liệu này, Multinomial Naive Bayes có khả năng phân loại tốt hơn, đặc biệt là trong việc nhận diện 1 (True/positive).
  - Bernoulli Naive Bayes vượt trội ở chỉ số precision của 1 (True/positive) và có recall cao cho 0 (False/negative), nhưng bị hạn chế ở khả năng phát hiện các mẫu 1 với recall thấp.
  - Hiệu suất tổng thể và sự cân bằng giữa các cách tính hiệu suất, Multinomial Naive Bayes là sự lựa chọn tốt hơn
