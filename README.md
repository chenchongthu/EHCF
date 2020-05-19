# EHCF

This is our implementation of the paper:

*Chong Chen, Min Zhang, Weizhi Ma, Yongfeng Zhang, Yiqun Liu and Shaoping Ma. 2020. [Efﬁcient Heterogeneous Collaborative Filtering without Negative Sampling for Recommendation.](https://chenchongthu.github.io/files/AAAI_EHCF.pdf) 
In AAAI'20.*

**Please cite our AAAI'20 paper if you use our codes. Thanks!**

```
@inproceedings{chen2020efficient,
  title={Efﬁcient Heterogeneous Collaborative Filtering without Negative Sampling for Recommendation},
  author={Chen, Chong and Zhang, Min and Ma, Weizhi and Zhang, Yongfeng and Liu, Yiqun and Ma, Shaoping},
  booktitle={Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020},
}
```

Author: Chong Chen (cstchenc@163.com)

## Environments

- python
- Tensorflow
- numpy
- pandas

## Parameter settings

The parameters for Beibei datasets is:

self.weight = [0.1, 0.1, 0.1]

self.coefficient = [0.05, 0.8, 0.15]

The parameters for Taobao datasets is:

self.weight = [0.01, 0.01, 0.01]

self.coefficient = [1.0/6, 4.0/6, 1.0/6]

## Corrections 

We would like to correct one typo error of the paper. In table 2, the HR@10 and NDCG@10 results of our EHCF on Beibei dataset should be 0.1551 and 0.0831, which are much better than baseline methods.


## Example to run the codes		

Train and evaluate the model:

```
python EHCF.py
```

## Suggestions for parameters

Three important parameters need to be tuned for different datasets, which are:
```
self.weight = [0.1, 0.1, 0.1]
self.coefficient = [0.05, 0.8, 0.15]
deep.dropout_keep_prob: 0.5
```

Specifically, we suggest to tune "self.weight" among \[0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5]. It's also acceptable to simply make the three weights the same, e.g., self.weight = \[0.1, 0.1, 0.1] or self.weight = \[0.01, 0.01, 0.01]. Generally, this parameter is related to the sparsity of dataset. If the dataset is more sparse, then a small value of negative_weight may lead to a better performance.

The coefficient parameter determined the importance of different tasks in multi-task learning. In our datasets, there are three loss coefﬁcients λ 1 , λ 2 , and λ 3 . As λ 1 + λ 2 + λ 3 = 1, when λ 1 and λ 2 are given, the value of λ 3 is determined. We tune the three coefﬁcients in \[0, 1/6, 2/6, 3/6, 4/6, 5/6, 1].

The performance of our EHCF is **much better** than existing multi-behavior models like CMF, NMTR, and MB-GCN (SIGIR2020). You can also contact us if you can not tune the parameters properly. 



First Update Date: May 19, 2020
