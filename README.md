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


### The chain rule of probability ###
```math
P(w_1, w_2, w_3,...w_n | Spam) = P(w_1 | Spam) \times P(w_2 |w1, Spam) \times P(w_3, | w_1, w_2, Spam) \times ...
```
- Xác suất của từ thứ 2 $(w_2)$ phụ thuộc vào việc từ thứ 1 xuất hiện.
- Xác suất của từ thứ 3 $(w_3)$ phụ thuộc vào cả $w_1$ và $w_2$.
- Ví dụ: $w_1$ là "Hồng", xác suất $w_2$ sẽ là "Kông" sẽ cao. Nếu $w_1$ là "Xe" thì xác suất $w_2$ là "Kông" sẽ thấp.

Vấn đề: Để tính được điều này, máy tính cần một lượng dữ liệu rất lớn để biết hết các tổ hợp có thể xảy ra 
=> Điều này rất phức tạp về mặt tính toán.

### Naive Bayes ###
Naive Bayes giả định rằng các từ xuất hiện độc lập với nhau. Khi đó, xác suất của cả câu sẽ bằng tích các xác suất của từng từ khoá:
```math
P(Words|Spam) \approx P(Word_1 | Spam) \times P(Word_2 | Spam) \times P(Word_3 | Spam) \times ... \times P(Word_n | Spam)
```
- $P(Kông|Hồng,Spam)$ sẽ đơn giản hoá thành $P(Kông | Spam)$ 
#### Tại sao gọi là "Naive"? ####
- Nó "naive" vì trong ngôn ngữ tự nhiên, giả định này là SAI. Các từ luôn đi đôi với nhau (New York, Hồng Kông, Machine Learning...).
- Mặc dù giả định này là sai về mặt ngữ pháp, nó lại hoạt động hiểu quả trong việc phân loại văn bản.

Ta có:
```math
P(Word | Spam) = \dfrac{\text{Số lần từ khoá (word) xuất hiện trong tập Spam}}{\text{Tổng số lượng từ có trong tập Spam}}
```
Công thức này có một lỗi đối với từ khoá chưa bao giờ xuất hiện trong tập huấn luyện. 
Ví dụ: 
- Tin nhắn mới có từ "Lions".
- Trong tập dữ liệu huấn luyện (Spam), từ "Lions" chưa xuất hiện nên $P(Lions | Spam) = 0$. 
- Vì là phép nhân $P(Word_1 \times Word_2 \times 0 \times ...)$, toàn bộ xác suất sẽ bằng 0.
- Tin nhắn sẽ bị phân là không phải Spam, dù các từ khác có xác suất cao.

Để giải quyết vấn đề ta sẽ sử dụng phương pháp Laplace Smoothing:
```math
P(Word_i | Spam) = \dfrac{Count(Word_i \text{ in spam}) + 1}{\text{Total Words in Spam} + \text{Vocab size}}
```
- Vocab size: Tổng số từ vựng độc nhất trong toàn bộ tập dữ liệu.

### Arithmetic Underflow ###

<h2>🛠️ Installation Steps:</h2>

<p>1. Initialize Conda environment</p>

```
conda env create -f environment.yml
```

<p>2. Clone the repository</p>

```
git clone https://github.com/tavenguyen/spam-detection.git
```

