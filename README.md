# NNN — NumPy Neural Network
 
**English** | [Polski](#nnn--sieć-neuronowa-w-numpy-wersja-polska)
 
A small neural network library written from scratch in NumPy. It supports regression (single output), binary classification and multiclass classification.
 
The goal of the project is deep undersanding of how neural networks work under the hood: every forward pass, gradient and parameter update is derived by hand and implemented without deep learning frameworks.
 
> **Status:** work in progress. The core building blocks (layers, activations, losses, model, optimizers) are implemented and tested. The training loop and benchmarks are planned next (see [Roadmap](#roadmap)).
 
## Features
 
- **Dense layer:** He or Xavier initialization and L2 regularization
- **Activations:** ReLU, LeakyReLU, Tanh, Sigmoid and Softmax
- **Losses computed from logits:** MSE, binary cross-entropy (BCE), cross-entropy (CE)
- **Model:** A list of layers with `forward`, `backward`, `predict`, `predict_proba`, `params` and `l2_penalty` methods
- **Optimizers:** SGD with optional momentum, Adam with bias correction
- **Tests:** Each module has it's own set of tests. All of the tests are optimized to work for all variants of said module eg. MSE, BCE and CE in Losses.
  
The core uses **NumPy only**. No TensorFlow, PyTorch or Keras.
 
## Design decisions
 
### Shapes
 
| Array | Shape |
|---|---|
| Input `X` | `(n, d_in)` |
| Weights `W` | `(d_in, d_out)` |
| Bias `b` | `(1, d_out)` |
| Target `y` (regression, binary) | `(n, 1)` |
| Target `y` (multiclass, one-hot) | `(n, k)` |
 
### Explicit composition
 
Every network is assembled by hand: a list of layers, a separate loss and a separate optimizer. Each layer has `forward(...)` and `backward(...)` part. 
 
### Losses work on logits
 
The model always outputs raw logits $Z$. The output activation is fused with the loss:
 
| Task | Output activation | Loss |
|---|---|---|
| Regression | identity | MSE |
| Binary classification | sigmoid | BCE |
| Multiclass classification | softmax | CE |
 
For all three pairs, the gradient with respect to the logits has the same form:
 
$$\frac{\partial L}{\partial Z} = \frac{\hat{y} - y}{n}$$
 
where $\hat{y}$ is the identity, sigmoid or softmax of $Z$. This is why MSE is defined with a factor of $\tfrac{1}{2}$:
 
$$L_{\text{MSE}} = \frac{1}{2n}\sum_{i}(\hat{y}_i - y_i)^2$$
 
Computing BCE and CE from logits also makes them numerically stable:
 
$$L_{\text{BCE}} = \frac{1}{n}\sum_i \left[\max(z_i, 0) + \log\left(1 + e^{-|z_i|}\right) - y_i z_i\right]$$
 
$$L_{\text{CE}} = \frac{1}{n}\sum_i \left[\log\sum_j e^{z_{ij}} - \sum_j y_{ij} z_{ij}\right]$$
 
The log-sum-exp is computed after subtracting the row maximum, so large logits do not overflow. Division by $n$ happens only in the loss.
 
### Dense layer
 
Forward pass:
 
$$Z = XW + b$$
 
Backward pass, given $dZ = \partial L / \partial Z$:
 
$$dW = X^\top dZ + \lambda W, \qquad db = \sum_{i} dZ_{i,:}, \qquad dX = dZ\, W^\top$$
 
### L2 regularization
 
L2 is part of the gradient, not of the optimizer. Each Dense layer adds $\lambda W$ to $dW$, and the model reports the matching penalty:
 
$$\text{penalty} = \sum_{\text{Dense layers}} \frac{\lambda}{2}\lVert W\rVert^2$$
 
Biases are not regularized. This is classic L2 regularization, not decoupled weight decay (AdamW), which is a deliberate choice.
 
### Initialization
 
- **He** (for ReLU): $W \sim \mathcal{N}\left(0, \tfrac{2}{d_{in}}\right)$
- **Xavier** (for tanh and sigmoid): $W \sim \mathcal{N}\left(0, \tfrac{2}{d_{in} + d_{out}}\right)$

A `np.random.default_rng` generator is passed to each layer, so results are reproducible.
 
### Optimizers
 
SGD with momentum $\mu$ (plain SGD when $\mu = 0$):
 
$$v \leftarrow \mu v + g, \qquad p \leftarrow p - \eta v$$
 
Adam with bias correction:
 
$$m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2$$
 
$$\hat{m} = \frac{m}{1-\beta_1^t}, \qquad \hat{v} = \frac{v}{1-\beta_2^t}, \qquad p \leftarrow p - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$
 
Optimizers update parameters in place, using the `(parameter, gradient)` pairs returned by `Model.params()`.
 
## Project structure
 
| File | Contents |
|---|---|
| `activations.py` | ReLU, LeakyReLU, Tanh, Sigmoid, Softmax |
| `layers.py` | `Dense` layer |
| `losses.py` | `MSE`, `BCE`, `CE` |
| `model.py` | `Model`: forward and backward passes, predictions, L2 penalty, parameters |
| `optimizers.py` | `SGD`, `Adam` |
| `gradcheck.py` | Helpers for numerical gradient checking |
| `test_*.py` | Tests for each module |
 
## Testing
 
Every activation, layer and loss is verified with a **numerical gradient check**. The analytic gradient is compared to a central-difference estimate in `float64`, and the relative error
 
$$\frac{\lVert g_{\text{analytic}} - g_{\text{numeric}}\rVert}{\lVert g_{\text{analytic}}\rVert + \lVert g_{\text{numeric}}\rVert}$$
 
must be below $10^{-6}$. Tests also cover shapes, call order, hand-computed examples and extreme input values.
 
The tests are plain Python scripts:
 
```bash
python test_activations.py
python test_layers.py
python test_losses.py
python test_model.py
python test_optimizers.py
```
 
## Roadmap
 
- [x] Activations and gradient check
- [x] Dense layer
- [x] Losses: MSE, BCE, CE
- [x] Model
- [x] Optimizers: SGD, momentum, Adam
- [ ] Training loop: mini-batches, shuffling, validation, early stopping
- [ ] Validation and comparison with scikit-learn's `MLPRegressor` and `MLPClassifier`
- [ ] Examples, plots, usage guide, `requirements.txt` and pytest
---
 
# NNN — sieć neuronowa w NumPy (wersja polska)
 
[English](#nnn--numpy-neural-network) | **Polski**
 
Mała biblioteka sieci neuronowych napisana od zera w NumPy. Obsługuje regresję (jedno wyjście), klasyfikację binarną i klasyfikację wieloklasową.
 
Celem projektu jest dogłębne zrozumienie działania sieci neuronowych od środka. Każde przejście w przód, każdy gradient i każda aktualizacja parametrów są wyprowadzone ręcznie i zaimplementowane bez frameworków do uczenia głębokiego.
 
> **Status:** projekt w trakcie realizacji. Podstawowe elementy (warstwy, aktywacje, straty, model, optymalizatory) są zaimplementowane i przetestowane. Kolejne w planie są pętla treningowa i porównania (zob. [Plan działania](#plan-działania)).
 
## Funkcje
 
- **Warstwa Dense:** Inicjalizowana za pomocą He lub Xavier oraz z regularyzacją L2
- **Aktywacje:** ReLU, LeakyReLU, Tanh, Sigmoid (stabilna numerycznie), Softmax
- **Straty liczone z logitów:** MSE, binarna entropia krzyżowa (BCE), entropia krzyżowa (CE)
- **Model:** Lista warstw z metodami `forward`, `backward`, `predict`, `predict_proba`, `params` i `l2_penalty`
- **Optymalizatory:** SGD z opcjonalnym momentum, Adam z korektą obciążenia
- **Testy:** Każdy z modułów posiada swój dedykowany zestaw testów. Wszystkie testwy zostały zoptymalizowane pod kątem 
działania z różnymi wariantami danego modułu np. MSE, BCE i CE dla strat

Cały kod storzony został **wyłącznie z NumPy**, bez TensorFlow, PyTorch i Keras.
 
## Decyzje projektowe
 
### Kształty
 
| Tablica | Kształt |
|---|---|
| Wejście `X` | `(n, d_in)` |
| Wagi `W` | `(d_in, d_out)` |
| Bias `b` | `(1, d_out)` |
| Cel `y` (regresja, klasyfikacja binarna) | `(n, 1)` |
| Cel `y` (wieloklasowa, one-hot) | `(n, k)` |
 
### Tworzenie sieci
 
Każda sieć jest składana ręcznie: lista warstw, strata i optymalizator tworzone są osobno. Każda warstwa zawiera w sobie część `forward(...)` i `backward(...)`.
 
### Straty działają na logitach
 
Model zawsze zwraca surowe logity $Z$. Aktywacja wyjściowa jest połączona ze stratą:
 
| Zadanie | Aktywacja wyjściowa | Strata |
|---|---|---|
| Regresja | tożsamość | MSE |
| Klasyfikacja binarna | sigmoida | BCE |
| Klasyfikacja wieloklasowa | softmax | CE |
 
Dla wszystkich trzech par gradient względem logitów ma tę samą postać:
 
$$\frac{\partial L}{\partial Z} = \frac{\hat{y} - y}{n}$$
 
gdzie $\hat{y}$ to tożsamość, sigmoida lub softmax z $Z$. Dlatego MSE jest zdefiniowane z czynnikiem $\tfrac{1}{2}$:
 
$$L_{\text{MSE}} = \frac{1}{2n}\sum_{i}(\hat{y}_i - y_i)^2$$
 
Liczenie BCE i CE z logitów zapewnia też stabilność numeryczną:
 
$$L_{\text{BCE}} = \frac{1}{n}\sum_i \left[\max(z_i, 0) + \log\left(1 + e^{-|z_i|}\right) - y_i z_i\right]$$
 
$$L_{\text{CE}} = \frac{1}{n}\sum_i \left[\log\sum_j e^{z_{ij}} - \sum_j y_{ij} z_{ij}\right]$$
 
Log-sum-exp jest liczony po odjęciu maksimum w wierszu, więc duże logity nie powodują przepełnienia. Dzielenie przez $n$ odbywa się tylko w stracie.
 
### Warstwa Dense
 
Przejście w przód:
 
$$Z = XW + b$$
 
Przejście wstecz, dla $dZ = \partial L / \partial Z$:
 
$$dW = X^\top dZ + \lambda W, \qquad db = \sum_{i} dZ_{i,:}, \qquad dX = dZ\, W^\top$$
 
### Regularyzacja L2
 
L2 jest częścią gradientu, a nie optymalizatora. Każda warstwa Dense dodaje $\lambda W$ do $dW$, a model zwraca odpowiadającą temu karę:
 
$$\text{kara} = \sum_{\text{warstwy Dense}} \frac{\lambda}{2}\lVert W\rVert^2$$
 
Biasy nie są regularyzowane. To klasyczna regularyzacja L2, a nie rozdzielony weight decay (AdamW). Wybór jest świadomy.
 
### Inicjalizacja
 
- **He** (dla ReLU): $W \sim \mathcal{N}\left(0, \tfrac{2}{d_{in}}\right)$
- **Xavier** (dla tanh i sigmoidy): $W \sim \mathcal{N}\left(0, \tfrac{2}{d_{in} + d_{out}}\right)$
Do każdej warstwy przekazywany jest generator `np.random.default_rng`, dzięki czemu wyniki są powtarzalne.
 
### Optymalizatory
 
SGD z momentum $\mu$ (zwykłe SGD, gdy $\mu = 0$):
 
$$v \leftarrow \mu v + g, \qquad p \leftarrow p - \eta v$$
 
Adam z korekcją obciążenia:
 
$$m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2$$
 
$$\hat{m} = \frac{m}{1-\beta_1^t}, \qquad \hat{v} = \frac{v}{1-\beta_2^t}, \qquad p \leftarrow p - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$
 
Optymalizatory aktualizują parametry w miejscu, na podstawie par `(parametr, gradient)` zwracanych przez `Model.params()`.
 
## Struktura projektu
 
| Plik | Zawartość |
|---|---|
| `activations.py` | ReLU, LeakyReLU, Tanh, Sigmoid, Softmax |
| `layers.py` | Warstwa `Dense` |
| `losses.py` | `MSE`, `BCE`, `CE` |
| `model.py` | `Model`: przejścia w przód i wstecz, predykcje, kara L2, parametry |
| `optimizers.py` | `SGD` (z momentum), `Adam` |
| `gradcheck.py` | Funkcje pomocnicze do numerycznego sprawdzania gradientów |
| `test_*.py` | Testy dla każdego modułu |
 
## Testy
 
Każda aktywacja, warstwa i strata przechodzi **numeryczne sprawdzenie gradientu**. Gradient analityczny jest porównywany z oszacowaniem różnicą centralną w `float64`, a błąd względny
 
$$\frac{\lVert g_{\text{analityczny}} - g_{\text{numeryczny}}\rVert}{\lVert g_{\text{analityczny}}\rVert + \lVert g_{\text{numeryczny}}\rVert}$$
 
musi być mniejszy niż $10^{-6}$. Testy sprawdzają też kształty, kolejność wywołań, przykłady policzone ręcznie i wartości skrajne.
 
Testy to zwykłe skrypty Pythona:
 
```bash
python test_activations.py
python test_layers.py
python test_losses.py
python test_model.py
python test_optimizers.py
```
 
## Plan działania
 
- [x] Aktywacje i sprawdzanie gradientów
- [x] Warstwa Dense
- [x] Straty: MSE, BCE, CE
- [x] Model
- [x] Optymalizatory: SGD, momentum, Adam
- [ ] Pętla treningowa: mini-batche, tasowanie, walidacja, early stopping
- [ ] Walidacja oraz porównanie z `MLPRegressor` i `MLPClassifier` ze scikit-learn
- [ ] Przykłady, wykresy, instrukcja użycia, `requirements.txt` i pytest
 