clearTestDataModel(){
    rm -f ../data/1KGPMLM/hap_test_*
}
inference(){
    clearTestDataModel
    python3 TaskFor1KGPMLMinf.py 0.4 $1 $1 $2 > ../../records/bertmlm/raw/$1-$2.log
}

rm -f ../data/1KGPMLM/hap_train_*
clearTestDataModel
python3 TaskFor1KGPMLM.py 0.4 512 512 512 > ../../records/bertmlm/raw/512-local.log

train_start=512
test_start=1024
while((${test_start} <= 2560))
do
    inference ${train_start} ${test_start}
    test_start=`expr ${test_start} + 512`
done

