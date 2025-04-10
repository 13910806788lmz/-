import pandas as pd

flag10 = True
flag30 = False

if flag10 == True:
    path0 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\1G_data\part-00000.parquet'
    df0 = pd.read_parquet(path0)  

    path1 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\1G_data\part-00001.parquet'
    df1 = pd.read_parquet(path1)  

    path2 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\1G_data\part-00002.parquet'
    df2 = pd.read_parquet(path2)  

    path3 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\1G_data\part-00003.parquet'
    df3 = pd.read_parquet(path3)  

    path4 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\1G_data\part-00004.parquet'
    df4 = pd.read_parquet(path4)  

    path5 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\1G_data\part-00005.parquet'
    df5 = pd.read_parquet(path5)  

    path6 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\1G_data\part-00006.parquet'
    df6 = pd.read_parquet(path6)  

    path7 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\1G_data\part-00007.parquet'
    df7 = pd.read_parquet(path7)  

    df_10G = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7], axis=0, sort=False, join='outer')
    print('数据整合完毕', len(df_10G))

if flag30 == True:
    path0 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00000.parquet'
    df0 = pd.read_parquet(path0)
    df0.drop_duplicates(inplace=True)
    path1 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00001.parquet'
    df1 = pd.read_parquet(path1)
    df1.drop_duplicates(inplace=True) 
    path2 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00002.parquet'
    df2 = pd.read_parquet(path2)
    df2.drop_duplicates(inplace=True)
    path3 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00003.parquet'
    df3 = pd.read_parquet(path3)
    df3.drop_duplicates(inplace=True)
    path4 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00004.parquet'
    df4 = pd.read_parquet(path4)
    df4.drop_duplicates(inplace=True)
    path5 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00005.parquet'
    df5 = pd.read_parquet(path5)
    df5.drop_duplicates(inplace=True) 
    path6 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00006.parquet'
    df6 = pd.read_parquet(path6)
    df6.drop_duplicates(inplace=True)
    path7 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00007.parquet'
    df7 = pd.read_parquet(path7)
    df7.drop_duplicates(inplace=True) 
    path8 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00008.parquet'
    df8 = pd.read_parquet(path8)
    df8.drop_duplicates(inplace=True)
    path9 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00009.parquet'
    df9 = pd.read_parquet(path9)
    df9.drop_duplicates(inplace=True) 
    path10 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00010.parquet'
    df10 = pd.read_parquet(path10)
    df10.drop_duplicates(inplace=True)
    path11 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00011.parquet'
    df11 = pd.read_parquet(path11)
    df11.drop_duplicates(inplace=True)
    path12 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00012.parquet'
    df12 = pd.read_parquet(path12)
    df12.drop_duplicates(inplace=True)
    path13 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00013.parquet'
    df13 = pd.read_parquet(path13)
    df13.drop_duplicates(inplace=True)
    path14 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00014.parquet'
    df14 = pd.read_parquet(path14)
    df14.drop_duplicates(inplace=True) 
    path15 = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data.torrent\30G_data\part-00015.parquet'
    df15 = pd.read_parquet(path15)
    df15.drop_duplicates(inplace=True)

    df_30G_pre = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7], axis=0, sort=False, join='outer')
    df_30G = pd.concat([df_30G_pre, df8, df9, df10, df11, df12, df13, df14, df15], axis=0, sort=False, join='outer')
    # df_30G = pd.concat([df0, df1], axis=0, sort=False, join='outer')
    print(len(df_30G))  