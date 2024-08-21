net_scale="l"
echo "trans $net_scale scale"
rm -rf /Users/zhangyuan/Desktop/netowrk_sim/routing/ini_dir/*
rm -rf /Users/zhangyuan/Desktop/netowrk_sim/routing/networks/*

cp -rf "ospf_update_dataset/omnet_file/$net_scale/ini_dir" /Users/zhangyuan/Desktop/netowrk_sim/routing
cp -rf "ospf_update_dataset/omnet_file/$net_scale/networks" /Users/zhangyuan/Desktop/netowrk_sim/routing