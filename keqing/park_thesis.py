from lyh.worker import *



def read_all_data(data_path, st_name, fig_pa_path, res_path, res_fit_path, log_label='off'):
    raw_data = pd.read_excel(data_path, sheet_name=st_name)
    if st_name == '社区公园':
        raw_data = raw_data.iloc[:-9, :]
    # print(raw_data.head())
    # print(raw_data.columns[2:])
    res = []
    res_fitted = []
    fitted_x = [(ii + 1) * 0.25 for ii in range(int(50/0.25))]
    for i, r in raw_data.iterrows():

        x = []
        y = []
        colm = raw_data.columns[2:]
        for col_nm in range(len(colm)):
            # 为啥在台式机上这个地方没有报错 奇怪
            if not pd.isna(raw_data[colm[col_nm]].iloc[i]):
                x.append(float(colm[col_nm]))
                y.append(raw_data[colm[col_nm]].iloc[i])
        # main_xy(x, y)
        fig_path = os.path.join(fig_pa_path, '{}_{}.jpg'.format(st_name, raw_data.iloc[i, 1]))
        if not os.path.exists(os.path.join(fig_pa_path, 'fitted_value')):
            os.mkdir(os.path.join(fig_pa_path, 'fitted_value'))
        fitted_file_path = os.path.join(os.path.join(fig_pa_path, 'fitted_value'),
                                        '{}_{}.xlsx'.format(st_name, raw_data.iloc[i, 1]))

        if len(x) > 10:
            tmp, fitted = main_xy_with_scale(x, y, fig_path, fitted_file_path, log_swith=log_label)
            res.append([raw_data.iloc[i, 1]] + tmp)
            res_fitted.append([raw_data.iloc[i, 1]] + list(fitted))

    pd.DataFrame(res, columns=['park', 'dis', 'formula', 'popt', 'R2']).to_csv(res_path, index=None, )
    pd.DataFrame(res_fitted, columns=['park'] + fitted_x).to_csv(res_fit_path, index=None, )

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
    # "综合公园"
    # "专类公园"
    # "社区公园"
    # "郊野公园"
    type_park = ["综合公园", "专类公园", "社区公园", "郊野公园"]
    for tp in type_park:
        read_all_data(r'D:\数据\keqing\parktype_距离衰减0426_每日_去长尾.xlsx',
                      st_name=tp,
                      fig_pa_path=r'D:\数据\keqing\res_log',
                      res_path=r'D:\数据\keqing\res_log\{}.csv'.format(tp),
                      res_fit_path=r'D:\数据\keqing\res_log\{}_fitted_value.csv'.format(tp),
                      log_label='on')

    # read_all_data_div(r'C:\Users\29420\Documents\WeChat Files\wxid_avb0egdv9lo422\FileStorage\File\2022-03\周末_工作日_parktype_距离衰减0330.xlsx',
    #                   st_name='周末',
    #                   fig_pa_path=r'E:\000数据\kq\div',
    #                   res_path=r'E:\000数据\kq\div\周末.csv')

    # res = []
    # all_data = read_all_data('path')
    # for data in all_data:
    #     res.append(main(data))
