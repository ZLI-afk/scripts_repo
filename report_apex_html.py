import os, sys, argparse, json, copy, traceback, glob
import numpy as np
from datetime import datetime
from monty.serialization import loadfn, dumpfn

HTML_HEAD = """
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            text-align: left;
        }
        
        .head1{
            font-size: 20px;
            font-weight: bold;
            white-space: pre-wrap;
            line-height: 2; 
            margin-top: 1rem;   
        }
        
        .head2{
            font-size: 18px;
            font-weight: bold;
            white-space: pre-wrap;
            line-height: 2;
        }
        
        .head3{
            font-size: 16px;
            font-weight: bold;
            white-space: pre-wrap;
            line-height: 2;
        }

        .tabletitle{
            font-size: 16px;
            font-weight: bold;
            word-wrap: break-word;
            line-height: 2;
            text-align: left;
        }
        
        .imagetitle{
            font-size: 16px;
            font-weight: bold;
            word-wrap: break-word;
            line-height: 2;
        }

        .doc {
            font-family: Verdana, sans-serif;
            text-align: left;
            display: inline-block;
            font-size: 16px;
            width: 100%;
            word-wrap: break-word;
            white-space: pre-wrap;
            line-height: 1.5;
            margin-bottom: 0.5rem;
        }
        
        #keys table {
          border-collapse: collapse;
          width: 100%;
        }
        #keys td {
          border: none;
          padding: 5px;
          text-align: left;
          line-height: 0.8;
        }
        
        img {
            max-width: 600px;
            max-height: 600px;
            height: auto;
            cursor: zoom-in;
        }

        .overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            overflow: auto;
        }

        .overlay img {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            cursor: zoom-out;
        }
    </style>
</head>
"""

def _init():
    global _global_dict
    _global_dict = {"START_TIME": datetime.now()}

def set_value(key, value):
    if key in _global_dict:
        print("WARNING: key %s has been used, the value will be modified from '" % key,_global_dict[key],"' to '",value,"'")
    _global_dict[key] = value

def get_value(key,defValue=None):
    return _global_dict.get(key,defValue)

def csv2table(csvfile):
    '''
    Transform a csv file to a table
    '''
    if not os.path.exists(csvfile):
        print(f"Error: {csvfile} does not exist!")
        return []
    with open(csvfile) as f:
        lines = f.readlines()
        table = []
        for line in lines:
            table.append(line.strip().split(","))
    return table

def dict2table(values):
    '''
    Transform a dict to a table
    The key should be the first column that is the example name, and the value should be a dict, which are the other columns, and the keys are the metric name.
    
    '''
    table = []
    metrics = []
    for k,v in values.items():
        for ik in v.keys():
            if ik not in metrics:
                metrics.append(ik)

    table.append(["example"] + metrics)
    for ikey in values.keys():
        table.append([ikey] + [values[ikey].get(imetric,None) for imetric in metrics])
    
    return table

def json2table(jsonfile):
    '''
    Transform a json file to a table
    The key should be the first column that is the example name, and the value should be a dict, which are the other columns, and the keys are the metric name.
    
    '''
    if not os.path.exists(jsonfile):
        print(f"Error: {jsonfile} does not exist!")
        return []
    values = json.load(open(jsonfile))
    table = []
    metrics = []
    for k,v in values.items():
        for ik in v.keys():
            if ik not in metrics:
                metrics.append(ik)

    table.append(["example"] + metrics)
    for ikey in values.keys():
        table.append([ikey] + [values[ikey].get(imetric,None) for imetric in metrics])
    
    return table

def json2table_sm(jsonfile):
    '''
    Transform a json file to a table
    The key should be the first column that is supermetrics name, and the value is the value.
    '''
    if not os.path.exists(jsonfile):
        print(f"Error: {jsonfile} does not exist!")
        return []
    values = json.load(open(jsonfile))
    table = []
    for k,v in values.items():
        if isinstance(v,(int,float,str,bool,type(None))):
            table.append([k,v])
        else:
            print(f"Warning: type of the value of {k} is {type(v)}, which is not supported. Ignored.")
    
    return table

def file2table(metricfile):
    # based on the file type to read the file
    if not os.path.exists(metricfile):
        print(f"Error: {metricfile} does not exist!")
        return None
    
    filetype = os.path.splitext(metricfile)[1]
    if filetype == ".csv":
        try:
            table = csv2table(metricfile)
        except:
            traceback.print_exc()
            print(f"Error: transfer {metricfile} to table failed!")
            return None
    elif filetype == ".json":
        try:
            table = json2table(metricfile)
        except:
            traceback.print_exc()
            print(f"Error: transfer {metricfile} to table failed!")
            return None
    else:
        print(f"Error: file '{filetype}' of table is not supported!")
        return None
    return table

def rotate_table(table):
    '''
    Rotate a table
    '''
    new_table = []
    for i in range(len(table[0])):
        new_table.append([table[j][i] for j in range(len(table))])
    return new_table

def isort(itable_input,head_list):
    itable = copy.deepcopy(itable_input)
    heads = copy.deepcopy(itable[0])
    sort_idx = []
    for i in head_list:
        if i not in heads:
            print("isort() ERROR:",i,"is not a head of input table. Table head is:",heads)
            return itable
        else:
            sort_idx.append(heads.index(i))
    jtable = []
    for i in itable[1:]:
        jtable.append([i[j] for j in sort_idx] + [i])
    jtable.sort()
    return [heads] + [i[-1] for i in jtable]

def output_float(f, prec=4):
    '''
    Output a string of float with precision, default 2
    If f is a small number, output in scientific notation
    '''
    if f == None:
        return "---"
    elif isinstance(f, str):
        return f
    elif isinstance(f, int):
        return str(f)
    
    try:
        f = float(f)
    except:
        return f
    
    if abs(f) < pow(10, -1*prec):
        return '%.2e' % f
    else:
        return '%.*f' % (prec, f)
    
def judge_metric(x, criteria):
    '''
    Judge if a metric is good or bad based on criteria
    '''
    try:
        x = float(x)
        sm_pass = eval(criteria)
        return bool(sm_pass)
    except:
        return None
    
def format_table(table,metrics_name=None, sort=None, criteria=None, color={True:"green",False:"red"}):
    '''
    table: a list of list, each list is a row of the table. The first row is the head of the table
    metrics_name: a list of metrics name, which will be output in the table
    sort: a list of metrics name, which will be used to sort the table
    criteria: a dict of criteria, the key is the metric name, and the value is the criteria
    
    Do things:
    1. If the value is a number, then round it to 4 digits
    2. If the value pass the criteria, the value will be colored to green, else red.
    3. If the value is None, then transfer to "---"
    
    criteria is a dict:
    {
        "key": "key_name",
        "criteria": "x > 0"   # this is a string, should be evaluated by python
    }
    
    return a table, and a dict of pass number
    {
        "key_metrics": {"pass": 1, "notpass": 2},
        "all": {"pass": 3, "total": 4}
    }
    '''
    # print("criteria:",criteria)
    # print("table head:",table[0])
    
    if metrics_name in [[],None]:
        metrics_name = table[0]
    if table[0][0] not in metrics_name:
        metrics_name = [table[0][0]] + metrics_name # the first colume should be the example name
    
    metric_idx = []
    for i in metrics_name:
        if i not in table[0]:
            metric_idx.append(None)
        else:
            metric_idx.append(table[0].index(i))
    
    new_table = [metrics_name]
    for i in range(1,len(table)):
        new_table.append([None if j == None else table[i][j] for j in metric_idx])
    
    if sort:
        new_table = isort(new_table,sort)    
    table = new_table
        
    pass_num = {k:{"pass":0,"notpass":0} for k in criteria.keys()}
    pass_num["all"] = {"pass":0,"total":len(table)-1}
    for i in range(1,len(table)):
        allpass = True
        for j in range(len(table[i])):
            metric_name = table[0][j]
            if metric_name in criteria:
                metric_pass = judge_metric(table[i][j], criteria[metric_name])
                if metric_pass == True:
                    table[i][j] = '<font color="%s">%s</font>' % (color[True], output_float(table[i][j]))
                    pass_num[metric_name]["pass"] += 1
                else:
                    table[i][j] = '<font color="%s">%s</font>' % (color[False], output_float(table[i][j]))
                    if metric_pass == False:
                        pass_num[metric_name]["notpass"] += 1
                    allpass = False
            else:
                table[i][j] = output_float(table[i][j])
        if allpass:
            pass_num["all"]["pass"] += 1
                    
    return table, pass_num

def gen_criteria(criteria,pass_num):
    html =  f'''
    <table border="2px">
        <tbody>
            <tr>
                <td>Key</td>
                <td>Pass/Total</td>
                <td>Criteria</td>
            </tr>
        '''
    for key in criteria.keys():
        if key not in pass_num:
            continue
        icolor = "green" if pass_num[key]["pass"] == pass_num[key]["pass"]+pass_num[key]["notpass"] else "red"
        html += f'''<tr>
                <td>{key}</td>
                <td style="color:{icolor}">{pass_num[key]["pass"]}/{pass_num[key]["pass"]+pass_num[key]["notpass"]}</td> 
                <td>{criteria[key]}</td>
            </tr>
        '''
        
    html += '''</tbody>
    </table>\n\n'''
    return html

def gen_criteria_sm(criteria,sm):
    '''
    sm = {key: value}
    '''
    html =  f'''
    <table border="2px">
        <tbody>
            <tr>
                <td>Metric</td>
                <td>Value</td>
                <td>Criteria</td>
            </tr>
        '''
    for ik,iv in sm.items():
        if ik in criteria:
            icolor = "green" if judge_metric(iv, criteria[ik]) else "red"
            value = output_float(iv)
            html += f'''<tr>
                    <td>{ik}</td>
                    <td style="color:{icolor}">{value}</td> 
                    <td>{criteria[ik]}</td>
                </tr>
            '''
        else:
            html += f'''<tr>
                    <td>{ik}</td>
                    <td>{iv}</td> 
                    <td>---</td>
                </tr>
            '''
        
    html += '''</tbody>
    </table>\n\n'''
    return html

def _table2html(table,has_head=True):    
    # add title
    html = '''\t<table style="margin-left: 0; margin-right: auto;" border="2px">\n'''

    # add table head
    start_row = 0
    if has_head:
        html += '\t\t<thead><tr>'
        for i in range(len(table[0])):
            html += '<th>%s</th>' % table[0][i]
        html += '</tr></thead>\n'
        start_row = 1
    
    # add table body
    html += '\t\t<tbody>'
    for i in range(start_row,len(table)):
        html += '\t\t\t<tr>'
        for j in range(len(table[i])):
            html += '<td>%s</td>' % table[i][j]
        html += '</tr>\n'
    html += '\t\t</tbody>\n'
    html += '\t</table>\n'
    
    return html
    
def metrics2html(metrics_set):
    metric = metrics_set.get("content",{})
    criteria = metrics_set.get("criteria",{})
    title = metrics_set.get("title","")
    metrics = metrics_set.get("metrics",[])
    sort = metrics_set.get("sort",[])
    center = metrics_set.get("center",True)
    
    #if not os.path.exists(metric_file):
    #    print(f"Error: {metric_file} does not exist!")
    #    return ""
    
    table = dict2table(metric)
    if not table:
        print(f"Error: transfer {metric} to table failed!")
        return ""
    if table in [[],None]:
        return ""
    if not sort: sort = [table[0][0]]
    table, pass_num = format_table(table, metrics, sort, criteria)
    
    html = ""
    if title:
        html += f'''\t<div class="tabletitle">{title}</div>\n'''
        
    if criteria:
        passnum = pass_num["all"]["pass"]
        totalnum = pass_num["all"]["total"]
        icolor = "green" if passnum == totalnum else "red"
        # if all passed, then color the number to green, else red
        #html += f'''\t<div class="head2">Pass/Total: <font color="{icolor}">{passnum}/{totalnum} ({passnum/totalnum*100:.2f}%)</font></div>\n'''
        #html += gen_criteria(criteria,pass_num)
    html += _table2html(table,has_head=True)
    
    if center:
        html = "\t<center>\n" + html + "\t</center>\n"
    return html

def supermetrics2html(supermetrics_set):
    metric_file = supermetrics_set.get("content","")
    criteria = supermetrics_set.get("criteria",{})
    title = supermetrics_set.get("title","")
    center = supermetrics_set.get("center",True)
    
    if not os.path.exists(metric_file):
        print(f"Error: {metric_file} does not exist!")
        return ""
    
    sm = json.load(open(metric_file))
    if not sm:
        print("Error: read supermetrics failed!")
        return ""
    
    html = ""
    if title:
        html += f'''\t<div class="tabletitle">{title}</div>\n'''
    html += gen_criteria_sm(criteria,sm)
    if center:
        html = "\t<center>\n" + html + "\t</center>\n"
    return html

def table2html(table_set):
    filename = table_set.get("content","")
    title = table_set.get("title","")
    center = table_set.get("center",True)
    
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist!")
        return ""
    
    html = ""
    if title:
        html += f'''\t<div class="tabletitle\">{title}</div>\n'''
        
    filetype = os.path.splitext(filename)[1]
    if filetype == ".csv":
        table = csv2table(filename)
    else:
        print(f"Error: file type '{filetype}' of table is not supported!")
        return ""

    html += _table2html(table,has_head=True)
    if center:
        html = "\t<center>\n" + html + "\t</center>\n"
    return html

def image2html(image_set):
    image_file = image_set.get("content","")
    title = image_set.get("title","")
    center = image_set.get("center",True)
    
    if isinstance(image_file,str):
        image_file = [image_file]
        
    html = ""

    for ifile in image_file:
        if not os.path.exists(ifile):
            print(f"Error: image '{ifile}' does not exist!")
        html += f"""\t<img class="thumbnail" src="{ifile}" onclick="openFullscreen(this)">\n"""
    
    if title != "":
        html += f'''\t<div class="imagetitle">{title}</div>\n'''
    if center:
        html = "\t<center>\n" + html + "\t</center>\n"
    return html
    
def text2html(text_set):
    '''
    Transform a text to html format
    '''
    text = text_set.get("content","")
    if isinstance(text, str):
        text = text.split("\n")
    elif isinstance(text, list):
        pass
    else:
        print(f"Error: the content of text should be str or list, but {type(text)} is given!")
        return ""
    html = ""
    for ic in text:    
        html += f'''\t<div class="doc">    {ic}</div>\n'''
    return html   

def get_job_address():
    job = get_value("JOB_ADDRESS","")
    if job != "":
        job = f'''<a href="{job}">link</a>'''
    return job

def get_version():
    if os.path.isfile("version.dat"):
        with open("version.dat") as f:
            version = f.read().strip()
    else:
        version = ""
    return version
    
def get_test_date():
    from datetime import datetime
    run_date = get_value("START_TIME",datetime.now()).strftime("%Y-%m-%d")
    return run_date
         
def keys2html(keys):
    html = ""
    html += """\t<table id="keys">\n"""
    comm_keys = ["test_date","version","job_address"]
    if not keys.get("job_address",""):
        keys["job_address"] = get_job_address()
    if not keys.get("version",""):
        keys["version"] = get_version()
    if not keys.get("test_date",""):
        keys["test_date"] = get_test_date()
        
    # write the comman keys
    for ikey in comm_keys:
        html += f'''\t\t<tr><td><strong>{ikey.capitalize()}</strong></td><td>:</td><td>{keys.get(ikey,"")}</td></tr>\n'''
    # write the other keys
    for ikey in keys.keys():
        if ikey not in comm_keys:
            html += f'''\t\t<tr><td><strong>{ikey.capitalize()}</strong></td><td>:</td><td>{keys.get(ikey,"")}</td></tr>\n'''
    html += """\t</table>\n"""
    return html

def gen_script(has_image=False):
    if not has_image:
        return ""
    
    html = ""
    
    if has_image:
        # add script for image zoom
        html += '''
    <div class="overlay" id="overlay" onclick="closeFullscreen()">
        <img id="fullscreenImage" src="">
    </div>
    '''
    
    html += '''\t<script>\n'''
    
    if has_image:
        html += '''
        var overlay = document.getElementById("overlay");
        var fullscreenImage = document.getElementById("fullscreenImage");
        
        function openFullscreen(imgElement) {
            fullscreenImage.src = imgElement.src; 
            overlay.style.display = "block";
        }
        
        function closeFullscreen() {
            overlay.style.display = "none";
            fullscreenImage.src = ""
        }
        '''
        
    html += '''
    </script>
    '''
    
    return html

def gen_html(report_setting, output):
    '''
    A report should be like this:
    Test Date/Version/Targets/Datasets/Properties/Criteria/Job Address:
    
    Content Section:
    '''
    
    '''
    Keys shold be a dict:
    {
        "test_date": "2020-01-01",
        "version": "1.0",
        "targets": "target1",
        "datasets": "dataset1",
        "properties": "property1",
        "criteria": "criteria1",
        "job_address": "job_address1"
    }
    
    The content section should has the following format:
    head1:
    head2:
    head3:
    text:
    image:
    table:
    metrics:
    supermetrics:
    
    If the content is a table, it can have an extra criteria to color the value, and if the value pass the criteria, the value will be colored to green, else red.
    If the content is a table or image, it can have an extra title.
    If the content is text, it can be a list of str, and each str will be a paragraph.
    
    Should be a list of dict:
    {
        "type": "head1",
        "content": "This is a head1"
    },
    {
        "type": "text",
        "content": "This is a text" or ["This is a text1","This is a text2"]
    }
    {
        "type": "image",
        "content": "image1.png",
        "title": "This is a image"
    },
    {
        "type": "table",
        "title": "This is a table",
        "content": "table1.csv"
    },
    {
        "type": "metrics",  # the row of a metrics table should be the example name, and the column should be the metric name
        "title": "This is a metrics table",
        "content": "table1.csv"
        "criteria":{
            "key1": "x > 0",
            "key2": "x < 1", # key is the name of the column, and the criteria is a string, should be evaluated by python
        },
        "sort": ["key1","key2"] # sort the table based on the key, default is the first column
        "metrics": ["metric1","metric2",] # only show the metrics in the list, default is all
    },
    {
        "type": "supermetrics",  # the row of a metrics table should be the example name, and the column should be the metric name
        "title": "This is a metrics table",
        "content": "table1.csv"
        "criteria":{
            "key1": "x > 0",
            "key2": "x < 1", # key is the name of the column, and the criteria is a string, should be evaluated by python
        }
    }
    '''
    
    keys = report_setting.get("keys",{})
    content = report_setting.get("content",[])
    
    html = HTML_HEAD + "\n<body>\n"
    html += keys2html(keys) + "\n"

    # write content
    has_image = False
    for item in content:
        itype = item.get("type","text")
        icontent = item.get("content","")
        if itype in ["head1","head2","head3"]:
            html += f'''\t<div class="{itype}">{icontent}</div>\n'''
        elif itype == "text":
            html += text2html(item)
        elif itype == "image":
            html += image2html(item)
            has_image = True
        elif itype == "table":
            html += table2html(item)
        elif itype == "metrics":
            html += metrics2html(item)
        elif itype == "supermetrics":
            html += supermetrics2html(item)
        else:
            continue
        html += "\n"
    
    # add script for image zoom
    html += gen_script(has_image)
         
    html += """\n</body>\n</html>"""
    
    with open(output,"w") as f:
        f.write(html)
    
    return html

def ReportArgs(parser):  
    parser.description = "Read metrics.json and generate the report"
    parser.add_argument('-p', '--param', type=str, help='the parameter file, should be .json type', required=True)
    parser.add_argument('-o', '--output', type=str,  default="abacustest.html", help='The output file name, default is abacustest.html')
    return parser

def Report(all_dict: dict):
    _init()
    report_setting = all_dict.get("report", {})
    if report_setting == {}:
        print("Error: report section is empty!")
        sys.exit(1)

    output = "results.html"
    gen_html(report_setting, output)

def simplify_paths(path_list: list) -> dict:
    # only one path, return it with only basename
    if len(path_list) == 1:
        return {path_list[0]: '.../' + os.path.basename(path_list[0])}
    else:
        # Split all paths into components
        split_paths = [os.path.normpath(p).split(os.sep) for p in path_list]

        # Find common prefix
        common_prefix = os.path.commonprefix(split_paths)
        common_prefix_len = len(common_prefix)

        # Remove common prefix from each path and create dictionary
        simplified_paths_dict = {
            os.sep.join(p): '.../' + os.sep.join(p[common_prefix_len:]) if common_prefix_len else os.sep.join(p)
            for p in split_paths
        }

        return simplified_paths_dict

def tag_dataset(orig_dataset: dict) -> dict:
    orig_work_path_list = [k for k in orig_dataset.keys()]
    try:
        simplified_path_dict = simplify_paths(orig_work_path_list)
        simplified_dataset = {simplified_path_dict[k]: v for k, v in orig_dataset.items()}
    except KeyError:
        simplified_dataset = orig_dataset
    # replace data id with tag specified in the dataset if exists
    tagged_dataset = {}
    for k, v in simplified_dataset.items():
        if tag := v.pop('tag', None):
            tagged_dataset[tag] = v
        else:
            tagged_dataset[k] = v
    return tagged_dataset

def cal_relative_error(predicted, actual):
    """
    predicted (float): predicted value
    actual (float): actual value
    return:
    float: relative error (%)
    """
    return abs(predicted - actual) / abs(actual)

def cal_STD(predicted: list, actual: list, point_group_sym: str):
    """
    categories based on point group symbols in DFT
    """
    if str(point_group_sym) in ['m-3m']:
        ela_pred_tensor = np.array([predicted[0][0], predicted[0][1], predicted[3][3]])
        ela_actu_tensor = np.array([actual[0][0], actual[0][1], actual[3][3]])
    elif str(point_group_sym) in ['6/mmm', 'mmm']:
        ela_pred_tensor = np.array([predicted[0][0], predicted[0][1], predicted[0][2], predicted[2][2], predicted[3][3]])
        ela_actu_tensor = np.array([actual[0][0], actual[0][1], actual[0][2], actual[2][2], actual[3][3]])
    elif str(point_group_sym) in ['4/mmm', '-3m']:
        ela_pred_tensor = np.array([predicted[0][0], predicted[0][1], predicted[0][2], predicted[2][2], predicted[3][3], predicted[5][5]])
        ela_actu_tensor = np.array([actual[0][0], actual[0][1], actual[0][2], actual[2][2], actual[3][3], actual[5][5]])
        
    STD_value = np.sqrt(np.sum(((ela_pred_tensor - ela_actu_tensor) / ela_actu_tensor) ** 2) / np.size(ela_actu_tensor))
    
    return STD_value

def prep_abc_content(orig_dict: dict, conf: str) -> dict:
    content_dict = {}
    idx = 3
    for k, v in orig_dict.items():
        new_dict = {k: None for k in METRICS_LIST}

        try:
            conf_info = v[conf]['relaxation']['structure_info']
            elastic_data = v[conf]['elastic_00']['result']
        except KeyError:
            print(f"{conf} information is not in {k}")
            continue
        else:
            tensor = elastic_data['elastic_tensor']
            new_dict["c11"] = tensor[0][0]
            new_dict["c12"] = tensor[0][1]
            new_dict["c13"] = tensor[0][2]
            new_dict["c33"] = tensor[2][2]
            new_dict["c44"] = tensor[3][3]
            new_dict["c66"] = tensor[5][5]
            new_dict["BV"] = elastic_data['BV']
            new_dict["GV"] = elastic_data['GV']

            # relative error of BV against experimental data
            if k != 'Expt':
                try:
                    new_dict["RE_BV_Expt"] = cal_relative_error(elastic_data['BV'], orig_dict['Expt'][conf]['elastic_00']['result']['BV'])
                except KeyError:
                    new_dict["RE_BV_Expt"] = None
                    print(f"Experimental information is not in {conf} for calculating RE_BV_Expt of {k}")
                except TypeError:
                    new_dict["RE_BV_Expt"] = None
                    print(f"Experimental infomation BV is not complete in {conf} for calculating RE_BV_Expt of {k}")
            # relative error of BV against DFT data
            if k != 'Expt' and k != 'DFT(abacus)':
                try:
                    new_dict["RE_BV_DFT"] = cal_relative_error(elastic_data['BV'], orig_dict['DFT(abacus)'][conf]['elastic_00']['result']['BV'])
                except KeyError:
                    new_dict["RE_BV_DFT"] = None
                    print(f"DFT information is not in {conf} for calculating RE_BV_DFT of {k}")
            # relative error of GV against experimental data
            if k != 'Expt':
                try:
                    new_dict["RE_GV_Expt"] = cal_relative_error(elastic_data['GV'], orig_dict['Expt'][conf]['elastic_00']['result']['GV'])
                except KeyError:
                    new_dict["RE_GV_Expt"] = None
                    print(f"Experimental information is not in {conf} for calculating RE_GV_Expt of {k}")
                except TypeError:
                    new_dict["RE_GV_Expt"] = None
                    print(f"Experimental infomation GV is not complete in {conf} for calculating RE_GV_Expt of {k}")
            # relative error of GV against DFT data
            if k != 'Expt' and k != 'DFT(abacus)':
                try:
                    new_dict["RE_GV_DFT"] = cal_relative_error(elastic_data['GV'], orig_dict['DFT(abacus)'][conf]['elastic_00']['result']['GV'])
                except KeyError:
                    new_dict["RE_GV_DFT"] = None
                    print(f"DFT information is not in {conf} for calculating RE_GV_DFT of {k}")

            # STD based on Expt data
            if k != 'Expt':
                try:
                    new_dict["STD_Expt"] = cal_STD(tensor, orig_dict['Expt'][conf]['elastic_00']['result']['elastic_tensor'], \
                                                   orig_dict['DFT(abacus)'][conf]['relaxation']['structure_info']['point_group_symbol'])
                except KeyError:
                    new_dict["STD_Expt"] = None
                    print(f"Experimental information is not in {conf} for calculating STD_Expt of {k}")
                except TypeError:
                    new_dict["STD_Expt"] = None
                    print(f"Experimental information cij is not complete in {conf} for calculating STD_Expt of {k}")
            
            # STD based on DFT data
            if k != 'Expt' and k != 'DFT(abacus)':
                try:
                    new_dict["STD_DFT"] = cal_STD(tensor, orig_dict['DFT(abacus)'][conf]['elastic_00']['result']['elastic_tensor'], \
                                                  orig_dict['DFT(abacus)'][conf]['relaxation']['structure_info']['point_group_symbol'])
                except KeyError:
                    new_dict["STD_DFT"] = None
                    print(f"DFT information is not in {conf} for calculating STD_DFT of {k}")

            content_dict[k] = new_dict
        
        if k == 'Expt':
            new_dict["idx"] = 0
        elif k == 'DFT(abacus)':
            new_dict["idx"] = 1
        elif k == 'single-dai':
            new_dict["idx"] = 2
        elif k == 'mace':
            new_dict["idx"] = 3
        else:
            idx += 1
            new_dict["idx"] = idx

    return content_dict

def prep_abc_dict(orig_dict: dict) -> dict:
    all_confs = set()
    all_props = set()
    abc_all_dict ={
        "report": {
            "content": []
        }
    }
    for w in orig_dict.values():
        all_confs.update(w.keys())
        for c in w.values():
            all_props.update(c.keys())

    abc_confs_dict_list = []
    all_confs_list = list(all_confs)
    all_confs_list.sort()
    for conf in all_confs_list:
        # print(orig_dict['DFT(abacus)'][conf]['relaxation']['structure_info']['point_group_symbol'])
        conf_dict = {
            "type": "metrics",
            "content": prep_abc_content(orig_dict, conf),
            "title": conf,
            "criteria": {
                "RE_BV_Expt": "abs(x) < 0.2",
                "RE_GV_Expt": "abs(x) < 0.2",
                "RE_BV_DFT": "abs(x) < 0.2",
                "RE_GV_DFT": "abs(x) < 0.2",
                "STD_Expt": "abs(x) < 0.2",
                "STD_DFT": "abs(x) < 0.2",
            },
            "sort": ["idx"],
            "metrics": METRICS_LIST,
        }
        abc_confs_dict_list.append(conf_dict)
    
    abc_all_dict["report"]["content"] = abc_confs_dict_list

    return abc_all_dict

def eval_STD(orig_dict: dict):
    THRESHOLD = 0.2
    content_dict1 = {}

    content = orig_dict["report"]["content"]
    idx = 0
    for k in content[0]["content"].keys():
        if k != 'Expt' and k != 'DFT(abacus)':
            new_dict1 = {k: 0 for k in METRICS_LIST1}
            content_dict1[k] = new_dict1
            content_dict1[k]["idx"] = idx
            idx += 1
    
    for item in content:
        icontent = item.get("content","")
        for k, v in icontent.items():
            if k != 'Expt' and k != 'DFT(abacus)':
                try:
                    content_dict1[k]["Aver_STD_Expt/DFT"] += v["STD_Expt"]
                    if v["STD_Expt"] < THRESHOLD:
                        content_dict1[k]["STD_Expt/DFT_pass_num"] += 1
                except TypeError:
                    content_dict1[k]["Aver_STD_Expt/DFT"] += v["STD_DFT"]
                    if v["STD_DFT"] < THRESHOLD:
                        content_dict1[k]["STD_Expt/DFT_pass_num"] += 1
                content_dict1[k]["Aver_STD_DFT"] += v["STD_DFT"]
                if v["STD_DFT"] < THRESHOLD:
                    content_dict1[k]["STD_DFT_pass_num"] += 1
    for k, v in content_dict1.items():
        v["Aver_STD_Expt/DFT"] = v["Aver_STD_Expt/DFT"] / len(content)
        v["Aver_STD_DFT"] = v["Aver_STD_DFT"] / len(content)

    eval_STD_inf = {
        "type": "metrics",
        "title": "Evaluation of models by STD values of cij (STD < 0.2)",
        "content": content_dict1,
        "sort": ["idx"],
        "metrics": METRICS_LIST1,
    }

    return eval_STD_inf

def prep_head():
    head_inf_1 = {
        "type": "head1",
        "content": "1. Introduction",
    }
    head_inf_2 = {
        "type": "head1",
        "content": "2. Results",
    }

    return head_inf_1, head_inf_2

def main():
    input_path_list = sys.argv[1:]
    path_list = []
    for ii in input_path_list:
        glob_list = glob.glob(os.path.abspath(ii))
        path_list.extend(glob_list)
        path_list.sort()

    if not path_list:
        raise RuntimeError('Invalid work path indicated. No path has been found!')

    file_path_list = []
    for jj in path_list:
        if os.path.isfile(jj):
            file_path_list.append(jj)
        else:
            raise FileNotFoundError(f'Invalid json file path provided: {jj}')

    if not file_path_list:
        raise FileNotFoundError(
            'all_result.json not exist!'
        )
    
    all_data_dict = {}
    for kk in file_path_list:
        data_dict = loadfn(kk)
        try:
            workdir_id = data_dict.pop('work_path')
            _ = data_dict.pop('archive_key')
        except KeyError:
            print(f'Invalid json for result archive, will skip: {kk}')
            continue
        else:
            all_data_dict[workdir_id] = data_dict

    # simplify the work path key for all datasets
    simplified_dataset = tag_dataset(all_data_dict)
    # prepare model predictions and evaluation parameters
    abc_dict = prep_abc_dict(simplified_dataset)
    # general evaluation of model prediction
    eval_STD_inf = eval_STD(abc_dict)
    abc_dict["report"]["content"].insert(0, eval_STD_inf)
    # add heads
    head_inf_1, head_inf_2 = prep_head()
    abc_dict["report"]["content"].insert(0, head_inf_2)
    abc_dict["report"]["content"].insert(0, head_inf_1)

    # with open('abc_dict.json', 'w') as json_file:
    #     json.dump(abc_dict, json_file, indent = 4)
    
    Report(abc_dict)

METRICS_LIST1 = ["idx", "STD_Expt/DFT_pass_num", "STD_DFT_pass_num", "Aver_STD_Expt/DFT", "Aver_STD_DFT"]
METRICS_LIST = ["idx", "c11", "c12", "c13", "c33", "c44", "c66", "BV", "GV", "RE_BV_Expt", "RE_BV_DFT", "RE_GV_Expt", "RE_GV_DFT", "STD_Expt", "STD_DFT"]
    
if __name__ == "__main__":
    main()