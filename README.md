<h1 align="center" id="title">Spam Detection Simple Model</h1>

<p align="center"><img src="https://shields.io/badge/python-3.10-blue" alt="shields"></p>

### Bayesian Conditional Probability ###
```math
P(Spam | Word) = \dfrac{P(Word | Spam)}{P(Word)} \cdot P(Spam)
```
- $P(Spam | Word)$: Xác suất để một email được phân loại là Spam với điều kiện từ khóa (Word) xuất hiện trong nội dung của nó.
- $P(Word | Spam)$: Xác suất để từ khóa (Word) xuất hiện nếu ta đã biết chắc chắn email đó là Spam.
- $P(Spam)$: Xác suất để một email bất kỳ là Spam.
- $P(Word)$: Tần suất xuất hiện của từ khóa (Word) trong toàn bộ tập dữ liệu.

Ta giả định rằng các từ xuất hiện độc lập với nhau. Khi đó, xác suất của cả câu sẽ bằng tích các xác suất của từng từ khoá:
```math
P(Words|Spam) \approx P(Word_1 | Spam) \times P(Word_2 | Spam) \times P(Word_3 | Spam) \times ... \times P(Word_n | Spam)
```

```math
P(Word | Spam) = \dfrac{\text{Số lần từ khoá (word) xuất hiện trong tập Spam}}{\text{Tổng số lượng từ có trong tập Spam}}
```
Công thức này có một lỗi đối với từ khoá chưa bao giờ xuất hiện trong tập huấn luyện. \
Ví dụ: 
- Tin nhắn mới có từ "Lions".
- Trong tập dữ liệu huấn luyện (Spam), từ "Lions" chưa xuất hiện nên $P(Lions | Spam) = 0$. \
- Vì là phép nhân $P(Word_1 \times Word_2 \times 0 \times ...)$, toàn bộ xác suất sẽ bằng 0.
- Tin nhắn sẽ bị phân là không phải Spam, dù các từ khác có xác suất cao.

Để giải quyết vấn đề ta sẽ sử dụng phương pháp Laplace Smoothing \
```math
P(Word_i | Spam) = \dfrac{Count(Word_i in spam) + 1}{Total Words in Spam + Vocab_size}
```
<h2>🛠️ Installation Steps:</h2>

<p>1. Initialize Conda environment</p>

```
conda env create -f environment.yml
```

<p>2. Clone the repository</p>

```
git clone https://github.com/tavenguyen/spam-detection.git
```
