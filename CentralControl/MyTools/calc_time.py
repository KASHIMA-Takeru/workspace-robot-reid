# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:58:35 2024

@author: ab19109
処理時間を計算して日時分秒で表示する
"""

def disp_ela(start, end):
    # =============================================================================
    # start, end (float): 処理開始時刻と終了時刻. timeモジュールで取得する．    
    # =============================================================================
    
    #経過時間(sec)
    elapsed = round(end - start)
    
    #日にち
    days = elapsed // (86400)
    #時間
    hours = (elapsed % 86400) // 3600
    #分
    minutes = ((elapsed % 86400) % 3600) // 60
    #秒
    seconds = ((elapsed % 86400) % 3600) % 60
    
    msg = ("Elapsed: {} days {} hours {} minutes {} seconds".format(days, hours, minutes, seconds))

    return msg
