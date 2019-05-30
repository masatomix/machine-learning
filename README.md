
### setup

```bash
$ python --version
Python 3.7.1
$ python -m venv ./venv
$ source ./venv/bin/activate
(venv) $
(venv) $ pip install numpy matplotlib PyQt5
```

参考: https://qiita.com/masatomix/items/03419c7ea10262da18f3

ディープラーニングの書籍のコードを使わせていただいていますが、それを取得するには以下。

そとだしの関数を使わせていただく際には

```bash
$ pwd
/.../xxx/machine-learning
$ curl https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/master/common/functions.py -O
$ curl https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/master/common/gradient.py -O
```

としてコードを取得し、実際のコード達からは


```python
from gradient import numerical_gradient
from functions import mean_squared_error
```

として使わせていただきます。
理解のためにも、できるだけ自前で実装したいと思ってはいますが、、。




