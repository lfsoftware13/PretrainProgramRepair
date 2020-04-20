from scripts.extract_loss_from_log import create_extract_loss_re, loss_rule, program_fix_rule, extract_info_from_file_each_line
import plotly.offline as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt


def extract_loss_line_figure_data(file_path):
    extract_loss_fn = create_extract_loss_re(loss_rule, info_num=2)
    extract_program_fix_fn = create_extract_loss_re(program_fix_rule, info_num=4)
    loss_res, fix_res = extract_info_from_file_each_line(file_path, [extract_loss_fn, extract_program_fix_fn])

    result = {'epoch': [], 'loss': [], 'position': [], 'output': []}

    for i in range(int(len(loss_res)/3)):
        e, l = loss_res[3*i+1]
        is_c, pos, out, all = fix_res[3*i+1]
        result['epoch'].append(e)
        result['loss'].append(l)
        result['position'].append(pos)
        result['output'].append(all)

    return result


def draw_line_figure(epoch, data_dict, title, y_axis, epoch_num=30):
    trace_list = []
    styles = ['', 'dash', 'dot']
    i = 0

    for name, loss in data_dict.items():
        if i > 0:
            line_dict = dict(width=3, dash=styles[i])
        else:
            line_dict = dict(width=3)
        i += 1
        trace = go.Scatter(
            x=epoch[:epoch_num],
            y=loss[:epoch_num],
            mode='lines',
            name=name,
            line=line_dict,
        )
        trace_list.append(trace)
    # data = go.Data(trace_list)
    data = trace_list

    layout = go.Layout(
        # title=dict(title=title, titlefont=dict(size=24)),
        title=title,
        font=dict(size=24),
                       xaxis=dict(title='Epoch', titlefont=dict(size=24)),  # 横轴坐标
                       yaxis=dict(title=y_axis, titlefont=dict(size=24)),  # 纵轴坐标
                       #  autorange=False, range=[0, 1.0], type='linear',
                       legend=dict(font=dict(family='sans-serif',size=24,color='black'))  # 图例位置
                        )

    fig = dict(data=data, layout=layout)
    py.plot(fig)


def loss_main(file_path_list, name_list):
    data_list = [extract_loss_line_figure_data(file_path) for file_path in file_path_list]
    loss_dict = {name: data['loss'] for data, name in zip(data_list, name_list)}
    output_dict = {name: data['output'] for data, name in zip(data_list, name_list)}
    draw_line_figure(data_list[0]['epoch'], loss_dict, title='Loss下降曲线', y_axis='Loss')
    draw_line_figure(data_list[0]['epoch'], output_dict, title='验证集准确度曲线', y_axis='Accuracy')


if __name__ == '__main__':
    file_path_list = [r'./pretrain_log/pretrain_encoder_sample_config7_pretrain45.log',
                      r'./pretrain_log/pretrain_encoder_sample_no_pretrain_config1.log',
                      r'./pretrain_log/pretrain_encoder_sample_config5_pretrain35.log',
                      ]
    name_list = ['Main Model', 'No Pre-train', 'BERT']
    loss_main(file_path_list, name_list)


