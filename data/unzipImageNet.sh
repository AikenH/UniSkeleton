# AikenH 2021 batch processing
# unzip those tar into correspond dir
# after comfirm we delete those zipfiles

for file in `ls *.tar`
do
    todir = `echo $file | cut -d"." -f1`
    mkdir $todir && tar -xvf $file -C $todir
    # 实际上这里应该加入判断，相应的文件数目到底是不是相同的，但是我不会
    rm -rv $file
done