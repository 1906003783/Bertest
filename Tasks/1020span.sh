rm -f ../cache/model_1KGPMLM.pt
rm -f ../data/1KGPMLM/hap_train_*
rm -f ../data/1KGPMLM/hap_test_*
python3 TaskFor1KGPMLM.py -ftrs 512 -ltrs 5120
cp -f ../cache/model_1KGPMLM.pt ./