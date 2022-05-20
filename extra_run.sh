# @AIKENHONG 2021 
# run the main process by sh thisfile.sh
# we can expand this to support new feature like apex or ddp

# echo "input the dataset you want to run at: ['cifar100', 'cifar10']"
# read dataset
# if [ $dataset = "cifar10" ]
# then 
#     echo "run cifar10"
#     config='cifar10_resnet18_best.yaml'
# elif [ $dataset = "cifar100" ]
# then
#     echo "run cifar100"
#     config='cifa100_resnet18_stable76.yaml'
# else
#     echo "run default"
#     config='basic_config.yaml'
# fi

# python main.py --cfg $config 

python main.py --cfg config/imb_config.yaml