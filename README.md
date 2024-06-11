# HandwritingRecognition - Nhận dạng chữ viết tay bằng CRNN
- [Giới thiệu](#angel-giới-thiệu)
- [Cài đặt](#gear-cài-đặt)
- [Dataset](#anchor-dataset)
- [Train and predict](#anchor-train-and-predict)

## :angel: Giới thiệu
- Bài toán đặt ra là chuyển những hình ảnh chữ viết tay tiếng Việt thành dữ liệu lưu trong tệp văn bản (text). Đại khái, input là một hình ảnh chứa một câu văn và output là những đoạn text xuất hiện trong hình ảnh đó.
- Cách tiếp cận ở đây là sử dụng thuật toán deep learning để giải quyết. Cụ thể, chúng ta sẽ cần một CNNs để trích xuất các đặc trưng của ảnh, và bởi vì đầu ra là chuỗi, nên ta nghĩ ngay đến cần RNN để xử lý. Ngoài ra còn cần sử dụng CTC netword để tính loss trong quá trình huấn luyện và thu được output theo thời gian.
  
![image](https://github.com/tandat17z/HUST_ML_HandwritingRecognition/assets/126872123/12689b65-a2dc-446b-b541-890e2cb7b13c)

## :gear: Cài đặt
(Đảm bảo rằng bạn đã thiết lập môi trường để chạy python và git)

- Clone từ GitHub:
    ```bash
    git clone https://github.com/tandat17z/HUST_IntroToSoftwareEngineering_BKvivu.git
    ```
- Tới thư mục làm việc của project. Ví dụ:
    ```bash
    cd D:\HUST_IntroToSoftwareEngineering_BKvivu
    ```
- Install các thư viện cần trong project:
    ```bash
    pip install -r requirements.txt
    ```
    
## :anchor: Dataset:
- Dữ liệu để train (test) là một foldler chứa 2 folder con là **img** và **label**.
- Trong đó:
  - **img** là tập các file hình ảnh về chữ viết tay (trên 1 dòng)
  - **label** là tập các file txt chứa nội dung về hình ảnh tương ứng.
(các file trong img và label tương ứng cần có tên giống nhau)
- Tham khảo: [train_data](https://github.com/tandat17z/HUST_ML_HandwritingRecognition/tree/branch2/data/train)
- 
## :anchor: Train and predict
- Training:
    ```bash
    python main.py --dstrain {train_folder} --dsval {val_folder} --savedir {save_folder} --nepochs {num} 
    ```
    
Để biết chi tiết các cấu hình khi train, ta có thể sử dụng lệnh sau:
    ```
    python main.py -h
    ```
    
- Predict:
  - Tham khảo trong file predict.ipynb. Các bước:
  - Kết quả:

  ![image](https://github.com/tandat17z/HUST_ML_HandwritingRecognition/assets/126872123/ad95c00f-e57f-476d-bbad-6c28e5fcf019)
