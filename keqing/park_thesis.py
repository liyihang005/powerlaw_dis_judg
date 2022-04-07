from lyh.worker import *



def read_all_data(data_path, st_name, fig_pa_path, res_path):
    raw_data = pd.read_excel(data_path, sheet_name=st_name)
    if st_name == '社区公园':
        raw_data = raw_data.iloc[:-9, :]
    # print(raw_data.head())
    # print(raw_data.columns[2:])
    res = []
    for i, r in raw_data.iterrows():

        x = []
        y = []
        for col_nm in raw_data.columns[2:]:
            if not pd.isna(r[col_nm]):
                x.append(float(col_nm))
                y.append(r[col_nm])
        # main_xy(x, y)
        fig_path = os.path.join(fig_pa_path, '{}_{}.jpg'.format(st_name, raw_data.iloc[i, 1]))
        if len(x) > 1:
            tmp = main_xy_with_scale(x, y, fig_path)
            res.append([raw_data.iloc[i, 1]] + tmp)
    pd.DataFrame(res, columns=['park', 'dis', 'formula', 'popt', 'R2']).to_csv(res_path, index=None, )


def read_all_data_div(data_path, st_name, fig_pa_path, res_path):
    raw_data = pd.read_excel(data_path, sheet_name=st_name)
    # print(raw_data.head())
    # print(raw_data.columns[2:])
    res = []
    for i, r in raw_data.iterrows():

        x = []
        y = []
        for col_nm in raw_data.columns[2:]:
            if not pd.isna(r[col_nm]):
                x.append(float(col_nm))
                y.append(r[col_nm])
        # main_xy(x, y)
        fig_path = os.path.join(fig_pa_path, '{}_{}.jpg'.format(st_name, raw_data.iloc[i, 1]))
        if len(x) > 10:
            try:
                tmp = main_xy_with_scale(x, y, fig_path)
                res.append([raw_data.iloc[i, 1]] + tmp)
            except:
                print(r)
    pd.DataFrame(res, columns=['row_idx', 'dis', 'formula', 'popt', 'R2']).to_csv(res_path, index=None, )


if __name__ == "__main__":
    read_all_data(r'C:\Users\29420\Documents\WeChat Files\wxid_avb0egdv9lo422\FileStorage\File\2022-03\park_type_距离衰减0329(1).xlsx',
                  st_name='社区公园',
                  fig_pa_path=r'E:\000数据\kq',
                  res_path=r'E:\000数据\kq\社区公园.csv')

    # read_all_data_div(r'C:\Users\29420\Documents\WeChat Files\wxid_avb0egdv9lo422\FileStorage\File\2022-03\周末_工作日_parktype_距离衰减0330.xlsx',
    #                   st_name='周末',
    #                   fig_pa_path=r'E:\000数据\kq\div',
    #                   res_path=r'E:\000数据\kq\div\周末.csv')

    # res = []
    # all_data = read_all_data('path')
    # for data in all_data:
    #     res.append(main(data))
