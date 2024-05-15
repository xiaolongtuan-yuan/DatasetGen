#!/bin/bash
current_user=$(whoami)
# 获取显示器的权限信息
xauthcode=$(sudo -u $SUDO_USER xauth list $DISPLAY)

# 授权为 root 用户显示信息
xauth add $xauthcode

echo "Permissions granted successfully."

conda deactivate
source venv/bin/activate
echo "Source venv successfully."

