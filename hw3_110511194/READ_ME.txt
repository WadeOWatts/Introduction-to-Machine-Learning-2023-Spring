在執行hw3.py前，需確任訓練圖片位於picture/train/裡，且測試圖片位於picture/test/裡。

執行過程中，會先使用Nu-SVM執行線性、徑向基函數與與sigmoid三種kernel type，接著再以C-SVM依序執行這三種，然後分別列印與測資比較後的正確率。

最後，選定C=3.0、RBF的模型，找出它的支持向量，並將其轉成圖片存於support_vectors/中。