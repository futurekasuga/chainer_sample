import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
import codecs

batch_size = 10                # バッチサイズ10
uses_device = -1

cp = np

# RNNの定義をするクラス
class Genarate_RNN(chainer.Chain):

	def __init__(self, n_words, nodes):
		super(Genarate_RNN, self).__init__()
		with self.init_scope():
			self.embed = L.EmbedID(n_words, n_words)
			self.l1 = L.LSTM(n_words, nodes)
			self.l2 = L.LSTM(nodes, nodes)
			self.l3 = L.Linear(nodes, n_words)

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def __call__(self, x):
		h0 = self.embed(x)
		h1 = self.l1(h0)
		h2 = self.l2(h1)
		y = self.l3(h2)
		return y

# カスタムUpdaterのクラス
class RNNUpdater(training.StandardUpdater):

	def __init__(self, train_iter, optimizer, device):
		super(RNNUpdater, self).__init__(
			train_iter,
			optimizer,
			device=device
		)

	def update_core(self):
		# 累積してゆく損失
		loss = 0
		
		# IteratorとOptimizerを取得
		train_iter = self.get_iterator('main')
		optimizer = self.get_optimizer('main')
		# ニューラルネットワークを取得
		model = optimizer.target
		
		# 文を一バッチ取得
		x = train_iter.__next__()

		# RNNのステータスをリセットする
		model.reset_state()

		# 分の長さだけ繰り返しRNNに学習
		for i in range(len(x[0])-1):
			# バッチ処理用の配列に
			batch = cp.array([s[i] for s in x], dtype=cp.int32)
			# 正解データ（次の文字）の配列
			t = cp.array([s[i+1] for s in x], dtype=cp.int32)
			# 全部が終端文字ならそれ以上学習する必要は無い
			if cp.min(batch) == 1 and cp.max(batch) == 1:
				break
			# 一つRNNを実行
			y = model(batch)
			# 結果との比較
			loss += F.softmax_cross_entropy(y, t)

		# 重みデータを一旦リセットする
		optimizer.target.cleargrads()
		# 誤差関数から逆伝播する
		loss.backward()
		# 新しい重みデータでアップデートする
		optimizer.update()

# ファイルを読み込む
s = codecs.open('all-sentences.txt', 'r', 'utf8')

# 全ての文
sentence = []

# 1行ずつ処理する
line = s.readline()
while line:
	# 一つの文
	one = [0] # 開始文字だけ
	# 行の中の単語を数字のリストにして追加
	one.extend(list(map(int,line.split(','))))
	# 行が終わったところで終端文字を入れる
	one.append(1)
	# 新しい文を追加
	sentence.append(one)
	line = s.readline()
s.close()

# 単語の種類
n_word = max([max(l) for l in sentence]) + 1

# 最長の文の長さ
l_max = max([len(l) for l in sentence])
# バッチ処理の都合で全て同じ長さに揃える必要がある
for i in range(len(sentence)):
	# 足りない長さは終端文字で埋める
	sentence[i].extend([1]*(l_max-len(sentence[i])))

# ニューラルネットワークの作成
model = Genarate_RNN(n_word, 200)

# 誤差逆伝播法アルゴリズムを選択
optimizer = optimizers.Adam()
optimizer.setup(model)

# Iteratorを作成
train_iter = iterators.SerialIterator(sentence, batch_size, shuffle=False)

# デバイスを選択してTrainerを作成する
updater = RNNUpdater(train_iter, optimizer, device=uses_device)
trainer = training.Trainer(updater, (30, 'epoch'), out="result")
# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar(update_interval=1))

# 機械学習を実行する
trainer.run()

# 学習結果を保存する
chainer.serializers.save_hdf5( 'chapt06.hdf5', model )
