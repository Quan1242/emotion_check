from src.data_preprocessing import load_data
from src.model import build_model, train_model
from src.evaluate import evaluate_model, plot_history

def main():
    # Đường dẫn đến thư mục train và test
    train_dir = 'data/train'
    test_dir = 'data/test'

    # Tải dữ liệu
    train_generator, test_generator = load_data(train_dir, test_dir)

    # Xây dựng và huấn luyện mô hình
    model = build_model()
    history = train_model(model, train_generator, test_generator)

    # Đánh giá và lưu kết quả
    evaluate_model(model, test_generator)
    plot_history(history)
    model.save('models/fer2013_model.h5')

if __name__ == '__main__':
    main()