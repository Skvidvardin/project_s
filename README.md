# Interpretable Neural Predictions with Differentiable Binary Variables

## Train model

**To train model:** run train.py

**Expected train process output:**

```
Classifier #params: 543905
IndependentLatentModel #params: 543002
Epoch 0, Iter 100, loss=1.586181082725525
Epoch 0, Iter 200, loss=1.5737664771080018
Epoch 0, Iter 300, loss=1.528490380048752
Epoch 0 iter 341: dev 0.3587647593093926
Epoch 1, Iter 400, loss=1.4608226644992828
Epoch 1, Iter 500, loss=1.4126615417003632
Epoch 1, Iter 600, loss=1.3822371792793273
Epoch 1 iter 682: dev 0.4141689373293241
Epoch 2, Iter 700, loss=1.3604095053672791
Epoch 2, Iter 800, loss=1.342703459262848
Epoch 2, Iter 900, loss=1.3697228884696961
Epoch 2, Iter 1000, loss=1.3594832921028137
Epoch 2 iter 1023: dev 0.4096276112621166
Epoch 3, Iter 1100, loss=1.3824577581882478
Epoch 3, Iter 1200, loss=1.4463294637203217
Epoch 3, Iter 1300, loss=1.4116907441616058
Epoch 3 iter 1364: dev 0.4114441416889996
Epoch 4, Iter 1400, loss=1.3880398976802826
Epoch 4, Iter 1500, loss=1.2905101680755615
Epoch 4, Iter 1600, loss=1.3231887793540955
Epoch 4, Iter 1700, loss=1.3187699580192567
Epoch 4 iter 1705: dev 0.4069028156217921
Epoch 5, Iter 1800, loss=1.2513631463050843
Epoch 5, Iter 1900, loss=1.3116234213113784
Epoch 5, Iter 2000, loss=1.292326488494873
Epoch 5 iter 2046: dev 0.4078110808352336
Epoch 6, Iter 2100, loss=1.301924568414688
Epoch 6, Iter 2200, loss=1.2765120899677276
Epoch 6, Iter 2300, loss=1.276368260383606
Epoch 6 iter 2387: dev 0.4196185286099731
Epoch 7, Iter 2400, loss=1.2658369106054306
Epoch 7, Iter 2500, loss=1.2897186541557313
```

## Gradients explosion

**Gradient exposion params:**

This is a reason why article is not reproducible without authors code and 
why we decided to use authors code after several unsuccessful attempts 
(truly, we were changing our code step-by-step and it was last thing that we could think may affect the results so much). 
It is interesting that there is no words in article about such strong restriction 
on weights initialization.

Without weights initialization gradients start to explode after 4 epoch (as it was shown in our report):

*(to reproduce comment line 52 in train.py file)*

```
def initialize_model_(model):

    for name, p in model.named_parameters():
        if "lstm" in name and len(p.shape) > 1:
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            torch.nn.init.constant_(p, 0.)
```

```
Classifier #params: 543905
IndependentLatentModel #params: 543002
Epoch 0, Iter 100, loss=1.5816270363330842
Epoch 0, Iter 200, loss=1.5509890520572662
Epoch 0, Iter 300, loss=1.5020290160179137
Epoch 0 iter 341: dev 0.3751135331513396
Epoch 1, Iter 400, loss=1.4009959709644317
Epoch 1, Iter 500, loss=1.3836878776550292
Epoch 1, Iter 600, loss=1.3642213428020478
Epoch 1 iter 682: dev 0.4032697547680261
Epoch 2, Iter 700, loss=1.3602299785614014
Epoch 2, Iter 800, loss=1.342182981967926
Epoch 2, Iter 900, loss=1.3607688128948212
Epoch 2, Iter 1000, loss=1.371349276304245
Epoch 2 iter 1023: dev 0.4187102633965316
Epoch 3, Iter 1100, loss=1.422239968776703
Epoch 3, Iter 1200, loss=1.5771746480464934
Epoch 3, Iter 1300, loss=1.8271769034862517
Epoch 3 iter 1364: dev 0.4305177111712711
Epoch 4, Iter 1400, loss=2.5464882028102873
Epoch 4, Iter 1500, loss=4.244617555141449
Epoch 4, Iter 1600, loss=7.264217414855957
Epoch 4, Iter 1700, loss=8.971223192214966
```

Moreover, even other type of xavier initialization leads to gradients explosion:

*to reproduce change line 418 in build_model.py file:*
```
from:
a = math.sqrt(3.0) * std 
to:
a = std
```


```
Classifier #params: 543905
IndependentLatentModel #params: 543002
Epoch 0, Iter 100, loss=1.5737767946720123
Epoch 0, Iter 200, loss=1.552616845369339
Epoch 0, Iter 300, loss=1.4835086345672608
Epoch 0 iter 341: dev 0.37783832879166407
Epoch 1, Iter 400, loss=1.4044223761558532
Epoch 1, Iter 500, loss=1.3612275075912477
Epoch 1, Iter 600, loss=1.360397081375122
Epoch 1 iter 682: dev 0.4277929155309466
Epoch 2, Iter 700, loss=1.3507706654071807
Epoch 2, Iter 800, loss=1.3328853285312652
Epoch 2, Iter 900, loss=1.3663746428489685
Epoch 2, Iter 1000, loss=1.3632111001014708
Epoch 2 iter 1023: dev 0.4368755676653616
Epoch 3, Iter 1100, loss=1.4302110707759856
Epoch 3, Iter 1200, loss=1.54404261469841
Epoch 3, Iter 1300, loss=1.816258397102356
Epoch 3 iter 1364: dev 0.4305177111712711
Epoch 4, Iter 1400, loss=2.483944935798645
Epoch 4, Iter 1500, loss=4.1073805809021
Epoch 4, Iter 1600, loss=6.933855652809143
Epoch 4, Iter 1700, loss=12.163096914291382
```

## Model inference

**To get model inference:** 

Run function **evaluate(model, data)** with **model** from sst_results/default/model.pt 
and read **data** from data folder by sst_reader.


