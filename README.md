# DANet_music2vec
DANetとmusic2vecを組み合わせた、音楽からのノイズ除去とノイズ除去された音楽のジャンル分類

## 概要
DANetとmusic2vecを組み合わせ、ノイズが混ざった音楽からノイズを除去し、さらにその音楽のジャンルを分類することを試みた。  
DANetとmusic2vecの実装や解説は、[DANetのリポジトリ](https://github.com/KMASAHIRO/DANet)と[music2vecのリポジトリ](https://github.com/KMASAHIRO/music2vec)を参照。  
まず環境音をノイズとして音楽に混ぜてそれらを分離するようにDANetを学習させ、事前に音楽ジャンル分類を学習させていたmusic2vecにDANetの出力を入力して音楽ジャンルの分類を行った。  
DANetとmusic2vecをつなげた推論とその結果の評価を実行したものを[Google Colaboratory](https://colab.research.google.com/drive/1oU9KUXhWjFCGsKjY259ed0wT20vYIrNB?usp=sharing)上にまとめている。

## DANetの学習
ノイズとして使用する環境音として[ESC-50](https://github.com/karolpiczak/ESC-50)、ジャンル分類する音楽として[GTZAN](http://opihi.cs.uvic.ca/sound/genres.tar.gz)を用いた。  
環境音1つを音楽に混ぜて分離させる学習と、環境音2つを音楽に混ぜて分離させる学習の2パターンを行った。  
環境音2つを混ぜる場合でも、出力は環境音と音楽の2つになるように学習させた。  

DANetでは訓練時にideal mask、推論時にkmeansを利用することでAttractorを作成して音声の分離を行うが、今回の学習では全20エポックの学習の内、最後の5エポックはideal maskではなくkmeansを用いてAttractorを作成した。  

シェルでDANet_kmeans1.pyを実行して環境音1つを混ぜた学習を行い、DANet_kmeans2.pyを実行して環境音2つを混ぜた学習を行った。  
DANet_kmeans_module.pyはそれらの実行時に利用した関数やモデルがまとめてある。

## music2vecの学習
[music2vecのリポジトリ](https://github.com/KMASAHIRO/music2vec)をまとめるときに作成・学習させた、事前学習済みSoundNetベースのモデルを利用した。  

## 処理の流れ
まず音楽データと環境音データをそれぞれ短時間フーリエ変換した後、DANetの入力サイズに合うようにそれぞれデータを分割してから混合してDANetに入力する。  
出力された音楽スペクトログラムの逆短時間フーリエ変換を行い、SoundNetの入力に合うようにリサンプリングした後、複数の出力を結合して元の音楽データに戻し、music2vecに入力して音楽ジャンルを分類させる。  
図で表すと下のようになる。  

<img width="500" alt="model_connection" src="https://user-images.githubusercontent.com/74399610/115969758-9edf6b00-a579-11eb-8976-6ad111082dec.png">

## 結果と評価
50個の音楽テストデータをジャンル分類させて混同行列を作成し、Accuracy・Precision・Recall・F1を求めた。  
比較として、DANetの出力をmusic2vecに入力する場合以外にも、元の音楽データをmusic2vecに入力する場合、DANetに入力される環境音と音楽との混合音声をmusic2vecに入力した場合にも同様のことを行って評価した。  
環境音を1つ混ぜた場合と2つ混ぜた場合それぞれについて下に示す。  

<img width="1000" alt="confusion_matrix1" src="https://user-images.githubusercontent.com/74399610/115970068-16fa6080-a57b-11eb-8ee6-83f695e98638.png">

<img width="1000" alt="confusion_matrix2" src="https://user-images.githubusercontent.com/74399610/115970070-19f55100-a57b-11eb-8022-ab9e350fea7d.png">

ここで、Precision・Recall・F1の値は各ラベルごとの値をデータ数に応じて重み付けをして平均したものである。例えば、下のような混同行列であれば下の計算式によりPrecisionはおよそ0.58になる。

<img width="500" alt="example_matrix" src="https://user-images.githubusercontent.com/74399610/115970185-d51dea00-a57b-11eb-9aba-3618abccf27c.png">

<img width="300" alt="example_matrix_precision" src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cfrac%7B10%7D%7B19%7D%5Ctimes%5Cfrac%7B15%7D%7B50%7D%2B%5Cfrac%7B12%7D%7B18%7D%5Ctimes%5Cfrac%7B25%7D%7B50%7D%2B%5Cfrac%7B6%7D%7B13%7D%5Ctimes%5Cfrac%7B10%7D%7B50%7D%5Cfallingdotseq0.58%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
\frac{10}{19}\times\frac{15}{50}+\frac{12}{18}\times\frac{25}{50}+\frac{6}{13}\times\frac{10}{50}\fallingdotseq0.58
\end{align*}
">

環境音2つを混ぜたときは環境音1つを混ぜたときよりも結果が悪くなっている。  
また、music2vecに環境音と音楽を混ぜたものを入力するよりもDANetの出力を入力したときの方が結果が悪くなっているので、DANetが思うように機能していないと言える。  
実際、環境音1つを混ぜたとき、DANet出力のGNSDRは-0.78、GSARは9.3、GSIRは6.0であり、環境音を2つ混ぜたときはGNSDRが0.76、GSARは9.4、GSIRは5.2であった。  
これは、カエルの鳴き声と風の音を分離したときの値(GNSDR:5.3 GSAR:12 GSIR:10)と比べても全体的に低く、特にGNSDRやGSIRが低くなっている。  
DANetから出力された音を聞いても分離されている感じはせず、大きな環境音が音楽に混ざったままであったり、逆に音楽が環境音とともに消されてしまったりしているような部分が多々あるように思えた。  
分類精度を上げるにはまずノイズ除去がきちんと行われるように工夫する必要がある。
