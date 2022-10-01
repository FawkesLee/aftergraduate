#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re
import time
from typing import Mapping


def modify_md_content(top):
    for root, dirs, files in os.walk(top, topdown=False):
        # 循环文件
        for file_name in files:
            file_name_split = file_name.split('.')
            try:
                if file_name_split[-1] == 'md':
                    print("处理文件", file_name_split[0])
                    # 找到md文件并且复制一份md文件路径
                    md_file_path = os.path.join(
                        root, '.'.join(file_name_split))
                    copy_md_file_path = os.path.join(root, '.'.join(
                        [f'{file_name_split[0]}_copy', file_name_split[1]]))

                    # 打开md文件然后进行替换
                    with open(md_file_path, 'r', encoding='utf8') as fr, open(copy_md_file_path, 'w', encoding='utf8') as fw:
                        data = fr.read()
                        # 选择md文件中想要替换的字段
                        data = re.sub(
                            'https://gitee.com/FawkesDoris/drawing-bed', 'https://github.com/wh21118310/drawing-bed', data)
                        fw.write(data)  # 新文件一次性写入原文件内容
                        # fw.flush()

                    # 删除原文件
                    os.remove(md_file_path)
                    # 重命名新文件名为原文件名
                    os.rename(copy_md_file_path, md_file_path)
                    print(f'{md_file_path} done...')
                    time.sleep(0.5)
            except FileNotFoundError as e:
                print(e)
        time.sleep(0.5)


if __name__ == '__main__':
    top = r'C:\Users\ABC\OneDrive\Documents\笔记'
    modify_md_content(top)
