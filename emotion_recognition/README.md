# Hướng dẫn chạy chương trình nhận diện biểu cảm

## Bước 1: Cài đặt Python 3.12
Đảm bảo rằng bạn đã cài đặt Python phiên bản 3.12 trên hệ thống của mình. Bạn có thể tải Python 3.12 từ [trang web chính thức của Python](https://www.python.org/downloads/).

## Bước 2: Cài đặt các thư viện cần thiết
Sử dụng lệnh sau để cài đặt tất cả các thư viện cần thiết:

```sh
pip install -r requirements.txt
```

## Bước 3: Tạo bộ dataset

```python
python3 build_dataset.py
```

## Bước 4: Huấn luyện mô hình. 

Lưu ý rằng bạn có thể thay đổi các tham số như checkpoints, experiment, total-epoch, và optimizer theo nhu cầu của bạn:

```python
python3 train_model.py --checkpoints checkpoints --experiment exp1 --total-epoch 75 --optimizer adam
```

## Bước 5: Đánh giá mô hình
Chạy tập lệnh test_recognizer.py để đánh giá mô hình đã huấn luyện. Thay path-to-checkpoints bằng đường dẫn tới mô hình đã lưu:


```python
python3 test_recognizer.py --model path-to-checkpoints
```

## Chạy chương trình nhận diện biểu cảm từ video
Để nhận diện biểu cảm từ video, sử dụng lệnh sau. Thay path-to-checkpoints bằng đường dẫn tới mô hình đã lưu và path-to-video bằng đường dẫn tới video cần nhận diện:

```python
python3 emotion_detector.py --model path-to-checkpoints --video path-to-video
```