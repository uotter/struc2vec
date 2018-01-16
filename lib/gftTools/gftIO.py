import pickle, zlib
import numpy as np
import pandas as pd
import binascii
import time
import random
import copy
import importlib


__version__ = '1.0.0'

def zdump(value, filename):
    dps = pickle.dumps(value, -1)
    zDps = zlib.compress(dps, 5)
    with open(filename, "wb", -1) as fpz:
        fpz.write(zDps)


def getCacheHeader():
    # type of python pickle
    header = int(7)
    header <<= 32
    # current version
    header += 1225
    return header


cacheHeader = getCacheHeader()


def loadMeta(byteArray):
    return pickle.loads(byteArray)


def getDataAndType(value):
    data = pickle.loads(value)
    type = checkDataType(data)
    return type, data


def zdump4CacheSever(value, timestamp, filename):
    dps = pickle.dumps(value, -1)
    if dps.__len__() > 1024 * 1024 * 1024 * 32:  # max of 32 G
        raise Exception("File:" + filename + "'s data is too large. data size:" + str(dps.__len__()))
    zDps = zlib.compress(dps, 5)
    if zDps.__len__() > 1024 * 1024 * 1024 * 4:  # max of 4G
        raise Exception("File:" + filename + "'s data is too large. file size:" + str(zDps.__len__()))
    with open(filename, "wb", -1) as os:
        # use little endian
        binHeader = cacheHeader.to_bytes(8, byteorder='little')
        timeBytes = timestamp.to_bytes(8, byteorder='little')
        os.write(binHeader)
        os.write(timeBytes)
        os.write(zDps)
        return 16 + zDps.__len__()


def zload4CacheServer(filename):
    with open(filename, "rb") as fpz:
        value = fpz.read()
    value = value[16:]   # drop the header and timestamp
    return pickle.loads(zlib.decompress(value), encoding='latin1')


def zload(filename):
    with open(filename, "rb") as fpz:
        value = fpz.read()
        try:
            ret = pickle.loads(zlib.decompress(value), encoding="latin1")
        except:
            ret = pickle.loads(value)
    return ret

inst_gid = None


def get_inst_gid():
    return inst_gid


# melt wide format -> long format
def convertMatrix2ColumnTab(matrix):
    if matrix.columns.size == 1:
        mat_copy = matrix.copy()
        col = oSet2Hex(mat_copy.columns.values)
        mat_copy.columns = col
        v_col_name = col[0]
        data_dic = {'T0': mat_copy.index, v_col_name: mat_copy[v_col_name].values}
        return pd.DataFrame(data_dic)
    matrix['idname'] = matrix.index
    return pd.melt(matrix, id_vars=['idname'])


def convertMatrix2ColumnTabWithName(matrix: pd.DataFrame, o_col_name='O0'):
    if matrix.columns.size == 1:
        mat_copy = matrix.copy()
        col = oSet2Hex(mat_copy.columns.values)
        mat_copy.columns = col
        v_col_name = col[0]
        data_dic = {'T0': mat_copy.index, v_col_name: mat_copy[v_col_name].values}
        return pd.DataFrame(data_dic)
    matrix[o_col_name] = matrix.index
    return pd.melt(matrix, id_vars=[o_col_name])


# pivot long format -> wide format
def convertColumnTabl2Matrix(columnTab):
    tName = None
    vName = None
    oName = None
    for colName in columnTab.columns:
        if (columnTab[colName].dtype == np.float64):
            if vName is None:
                vName = colName
        if istimestamparray(columnTab[colName]):
            if tName is None:
                tName = colName
        if (columnTab[colName].dtype == np.object):
            if oName is None:
                oName = colName

    if vName is None:
        raise Exception("v Name is None")

    if tName is None:
        raise Exception("tName is None")

    if oName is None:
        try:
            unique = columnTab.set_index(keys=tName, verify_integrity=True)
            unique.columns = np.chararray(1, itemsize=16, buffer='\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0')
            return unique
        except Exception:
            raise Exception("Can not transform TV to matrix, T is not unique")

    return columnTab.pivot_table(values=vName, index=tName, columns=oName)  ##将cov变成wide format格式


def isNonSymbol(df):
    if isinstance(df, pd.DataFrame):
        if ismatrix(df):
            return df.columns[0].__len__() == 0
    return False


# class saving data for both matrix(wide format) and columnTable(long format)
class GftTable:

    # add xarray for high dimension data.
    # about the added parameter matrixIsFromPython: in case matrixIsFromPython is not given, it will be None so the matrix will be transposed as before
    def __init__(self, matrix, columnTab, matrixIsFromPython, gid, columnOrders):
        # save matrix and column table.
        if (matrix is None):
            self.matrix = None
        elif matrixIsFromPython:
            self.matrix = matrix
        else:
            self.matrix = matrix.transpose()
        self.columnTab = columnTab
        self.type = "GftTable"
        self.gid = gid
        self.columnOrders = columnOrders

    def isNonSymbol(self):
        if self.matrix is not None:
            return self.matrix.columns[0].__len__() == 0
        return False

    @classmethod
    def fromCppMatrix(cls, matrix):
        return cls(matrix, None, False, None, None)

    @classmethod
    def fromCppMatrixWithGid(cls, matrix, gid):
        return cls(matrix, None, False, gid, None)

    @classmethod
    def fromPythonMatrix(cls, matrix):
        return cls(matrix, None, True, None, None)

    @classmethod
    def fromPythonMatrixWithGid(cls, matrix, gid):
        return cls(matrix, None, True, gid, None)

    @classmethod
    def fromColumnTable(cls, columnTable, columnOrders):
        return cls(None, columnTable, False, None, columnOrders)

    @classmethod
    def fromColumnTableWithGid(cls, columnTable, gid, columnOrders):
        return cls(None, columnTable, False, gid, columnOrders)

    @classmethod
    def fromBoth(cls, matrix, columnTable, matrixIsFromPython, columnOrders):
        return cls(matrix, columnTable, matrixIsFromPython, None, columnOrders)

    @classmethod
    def fromBothWithGid(cls, matrix, columnTable, matrixIsFromPython, gid, columnOrders):
        return cls(matrix, columnTable, matrixIsFromPython, gid, columnOrders)

    # const data, the return value should not be changed


    def asMatrix(self):
        if self.matrix is None:
            self.matrix = convertColumnTabl2Matrix(self.columnTab)
        return self.matrix

    def as_matrix(self):
        if self.matrix is None:
            self.matrix = convertColumnTabl2Matrix(self.columnTab)
        return self.matrix

    def as_mutable_matrix(self):
        ret = self.as_matrix()
        return ret.copy()

    # const data, the return value should not be changed

    def asColumnTab(self):
        if self.columnTab is None:
            self.columnTab = convertMatrix2ColumnTab(self.matrix)
        return self.columnTab

    def as_column_tab(self):
        if self.columnTab is None:
            self.columnTab = convertMatrix2ColumnTab(self.matrix)
        return self.columnTab

    def as_mutable_column_tab(self):
        ret = self.as_column_tab()
        return ret.copy()

    def asColumnTabWithName(self, id_name):
        if self.columnTab is None:
            self.columnTab = convertMatrix2ColumnTab(self.matrix, id_name)
        return self.columnTab


class gs_parameter:
    def __init__(self, gid):
        self.gid = gid

    @classmethod
    def gen_rand_para(cls):
        return gs_parameter(random.getrandbits(128).to_bytes(16, byteorder='little', signed=False))


class gs_context:
    def __init__(self, myself, input_gid_list, my_meta=None, input_meta_list=None):
        self.myself = myself
        self.input_list = input_gid_list
        self.my_meta = my_meta
        self.input_meta_list = input_meta_list

    # def __init__(self, myself, input_gid_list, begin_time, end_time, lookback, input_metas, my_meta):
    #     self.myself = myself
    #     self.input_list = input_gid_list
    #     self.begin_time = begin_time
    #     self.end_time = end_time
    #     self.lookback = lookback
    #     self.input_metas = input_metas
    #     self.my_meta = my_meta

    @classmethod
    def gen_rand_context(cls, input_len):
        gid = gs_parameter.gen_rand_para()
        input_list = [None] * input_len
        for i in range(input_len):
            input_list[i] = gs_parameter.gen_rand_para()
        return gs_context(gid, input_list, None, None)


def gen_gs_context_from_gid(my_gid, input_gid_list):
    my_para = gs_parameter(my_gid)
    input_para_list = []
    for gid in input_gid_list:
        input_para_list.append(gs_parameter(gid))
    return gs_context(my_para, input_para_list, None, None)


def gen_gs_context_with_meta(my_gid, input_gid_list, my_meta, input_meta_list):
    my_para = gs_parameter(my_gid)
    input_para_list = []
    for gid in input_gid_list:
        input_para_list.append(gs_parameter(gid))
    return gs_context(my_para, input_para_list, my_meta, input_meta_list)


def gen_gs_context_with_meta_and_dump_4_debug(my_gid, input_gid_list, my_meta, input_meta_list, file_path):
    ret = gen_gs_context_with_meta(my_gid, input_gid_list, my_meta, input_meta_list)
    zdump(ret, file_path)
    return gen_gs_context_with_meta(my_gid, input_gid_list, my_meta, input_meta_list)


def transformO4Input(ndInput):
    return np.chararray(shape=(ndInput.size), itemsize=16, buffer=ndInput.data)


# def transformO4Input(ndInput):
#     twoCol = np.ndarray(shape=(ndInput.size,2),dtype=np.uint64,buffer=ndInput.data)
#     return twoCol[:,0],twoCol[:,1]
oDataType = [('v1', np.uint64), ('v2', np.uint64)]


def transformOFromPandas2Output(ndOutput):
    ret = ndOutput.astype(oDataType)
    ret.dtype = np.complex128
    return ret


def transform2DataFrame(tableInput):
    oSet = transformO4Input(tableInput[0])
    tSet = transformTime4Input(tableInput[1])
    return pd.DataFrame(tableInput[2], index=oSet, columns=tSet)


# 62091 is the difference between 1970-01-01 and 1800-01-01
max = 0xFFFFFFFF00000000 + 62091

max_day_val = 100000
magic_null_day = -10000000
magic_null_day_dt = np.datetime64(magic_null_day, 'D')


def transformTime4Input(tArr):
    tArr.dtype = np.uint64
    tT = tArr - max
    tT.dtype = np.int64
    tT[tT > max_day_val] = magic_null_day  # if it large than the max possible value, than set it to some magic num. and than roll it back
    tT.dtype = 'datetime64[D]'
    tT[tT == magic_null_day_dt] = None
    return tT


def transformTime4Output(tArr):
    tDay = tArr.astype('datetime64[D]')
    tDay.dtype = np.uint64
    tDay = tDay + max
    return tDay


def transformOTInColumnTable(colTable):
    varDict = dict()
    for col in colTable:
        if col[1].dtype == np.complex128:
            varDict.__setitem__(col[0], transformO4Input(col[1]))
        elif col[1].dtype == np.int64 or col[1].dtype == np.uint64:
            varDict.__setitem__(col[0], transformTime4Input(col[1]))
        else:
            varDict.__setitem__(col[0], col[1])
    return varDict


# Martin: the correct name called from C++ code is transformInput, transformData is the old name for testing which will be obsoleted
PARAMETER_TYPE_ERROR = int(0)
PARAMETER_TYPE_STRING = int(1)
PARAMETER_TYPE_TIMESTAMP = int(2)
PARAMETER_TYPE_PANEL = int(3)
PARAMETER_TYPE_OBJECT_SET = int(5)
PARAMETER_TYPE_FREQUENCY = int(6)
PARAMETER_TYPE_PARAMETERDICT = int(15)
PARAMETER_TYPE_NUMBER_INT = int(20)
PARAMETER_TYPE_NUMBER_NUMRIC = int(21)
PARAMETER_TYPE_OnTnVn_TABLE = int(22)
PARAMETER_TYPE_PYTHON_ONLY = int(25)
PARAMETER_TYPE_UUID = int(4)


class PlaceHolder:
    def __init__(self, idx, bind_name):
        self.idx = idx
        self.bind_name = bind_name

class GsClosure:
    def __init__(self, owner_gid, func_gid, func, *binds):
        self.owner_gid = owner_gid
        self.func_gid = func_gid
        holder_count = 0
        self.args = list()
        self.func = func
        for bind in binds:
            if isinstance(bind, PlaceHolder):
                self.args.append(bind)
                holder_count += 1
            else:
                self.args.append(bind)
        self.holder_count = holder_count

    def get_owner_gid(self):
        raise self.owner_gid
    def get_func_gid(self):
        raise self.func_gid

    def get_placeholder_list(self):
        name_dict = dict()
        for arg in self.args:
            if isinstance(arg, PlaceHolder):
                name_dict[arg.idx] = arg.bind_name
        name_list = list()
        for i in range(name_dict.__len__()):
            name_list.append(name_dict[i+1])
        return name_list


    def call(self, *inputs):
        arg_size = len(inputs)
        if arg_size != self.holder_count:
            raise Exception("Lambda has {0} parameter(s), called with {1} parameter(s)".format(str(self.holder_count),
                                                                                               str(arg_size)))
        call_args = list()
        for arg in self.args:
            if isinstance(arg, PlaceHolder):
                call_args.append(wrap_gfttable_dataframe_with_gid(inputs[arg.idx - 1], None))
            else:
                call_args.append(arg)
        # "Please setup the func using the create_func_lambda() in function:[{0}]  ".format(self.func_gid)
        return self.func(*call_args)

    def __getstate__(self):
        attr_dict = self.__dict__.copy()
        attr_dict['func'] = None
        return attr_dict

    def gen_reassemble_code(self, closure_name, file_name, method_name):
        arg_names = list()
        for arg in self.args:
            if isinstance(arg, PlaceHolder):
                arg_names.append(arg.bind_name)
        names = ', '.join(arg_names)
        s = ("import {0} as {1}\n"
            "{2}.func = {1}.create_func_lambda()\n"
            "{2}_closure = {2}\n"
            "{2} = lambda {3}: {2}_closure.call({3})").format(file_name,method_name,closure_name,'names')
        print(s)

    # before call this, be sure you get 'owner' directory is ready
    # it will return all the sub closures it depends on.
    def reassemble_code(self):
        package_name = 'code.{0}.{1}.{2}.{3}.{4}'.format(self.owner_gid[0:2],self.owner_gid[2:4],self.owner_gid[4:8],self.owner_gid[8:])
        lambda_code = importlib.import_module(self.func_gid, package_name)
        self.func = lambda_code.create_func_lambda()

    def find_all_sub_closures(self, add_into_list):
        for arg in self.args:
            if isinstance(arg, GsClosure):
                # so the sub closure will at the header of the list.
                add_into_list = arg.find_all_sub_closures(add_into_list)
                if add_into_list is None:
                    add_into_list = list()
                add_into_list.append(arg)
        return add_into_list

    def create_lambda(self):
        return eval("lambda {0}: self.call({0})".format(self.get_placeholder_list()))


def find_all_closures(arg):
    if isinstance(arg, GsClosure):
        add_into_list = list()
        arg.find_all_sub_closures(add_into_list)
        add_into_list.append(arg)
        return add_into_list
    return None


def get_column_type(col_table: pd.DataFrame, name: str):
    dName = str(col_table[name].dtype)
    if 'datetime64' in dName:
        return PARAMETER_TYPE_TIMESTAMP
    elif 'float' in dName:
        return PARAMETER_TYPE_NUMBER_NUMRIC
    else:
        return PARAMETER_TYPE_UUID


def get_columns_type_dict(col_table: pd.DataFrame):
    ret = dict()
    for name in col_table.columns:
        ret[name] = get_column_type(col_table, name)
    return ret


def istimestamparray(array):
    return 'datetime64' in str(array.dtype)


def ismatrix(df: pd.DataFrame):
    return istimestamparray(df.index)


def checkDataType(obj):
    if isinstance(obj, str):
        return PARAMETER_TYPE_STRING
    elif isinstance(obj, int):
        return PARAMETER_TYPE_NUMBER_INT
    elif isinstance(obj, float):
        return PARAMETER_TYPE_NUMBER_NUMRIC
    elif isinstance(obj, dict):
        return PARAMETER_TYPE_PARAMETERDICT
    elif isinstance(obj, pd.DataFrame):
        return getDataFrameType(obj)
    elif isinstance(obj, GftTable):
        if obj.matrix is not None:
            return PARAMETER_TYPE_PANEL
        elif obj.columnTab is not None:
            return PARAMETER_TYPE_OnTnVn_TABLE
        else:
            return PARAMETER_TYPE_ERROR
    elif isinstance(obj, pd.Timestamp):
        return PARAMETER_TYPE_TIMESTAMP
    elif np.isscalar(obj):
        return PARAMETER_TYPE_NUMBER_NUMRIC
    else:
        return PARAMETER_TYPE_PYTHON_ONLY


def getDataFrameType(obj: pd.DataFrame):
    # TODO: this standard needs to be upgraded
    if istimestamparray(obj.index):
        return PARAMETER_TYPE_PANEL
    else:
        return PARAMETER_TYPE_OnTnVn_TABLE


def gs_day_2_pd_dt(days):
    # 62091 is the difference between 1970-01-01 and 1800-01-01
    days = days - 62091
    return pd.to_datetime(days, unit='D')


# 62091 is the difference between 1970-01-01 and 1800-01-01
def pd_dt_2_gs_day(ts):
    return int(ts.tz_localize(tz="UTC").timestamp() / 86400) + 62091


def transformCppInput(data, parType):
    if parType == PARAMETER_TYPE_PANEL:
        data = transform2DataFrame(data)
        return GftTable.fromCppMatrix(data.copy())
    if parType == PARAMETER_TYPE_OnTnVn_TABLE:
        col_range = range(len(data))
        column_names = [data[i][0] for i in col_range]
        data_list = [data[i][1] for i in col_range]
        data = pd.DataFrame(data=data_list, columns=column_names)
        return GftTable.fromColumnTable(data, column_names)
    if parType == PARAMETER_TYPE_FREQUENCY:
        tArr = transformTime4Input(data)
        return pd.DatetimeIndex(data=tArr, copy=False)
    if parType == PARAMETER_TYPE_OBJECT_SET:
        return transformO4Input(data)
    if parType == PARAMETER_TYPE_TIMESTAMP:
        return gs_day_2_pd_dt(data)
    return data


def test_coord(l_val, r_val):
    return l_val.equals(r_val)


def test_dataframe(l_val: pd.DataFrame, r_val: pd.DataFrame):
    return l_val.equals(r_val)


def test_gft_table(l_val, r_val):
    if (l_val.matrix is not None) and (r_val.matrix is not None):
        if not test_dataframe(l_val.matrix, r_val.matrix):
            return False

    if (l_val.columnTab is not None) and (r_val.columnTab is not None):
        if not test_dataframe(l_val.columnTab, r_val.columnTab):
            return False
    return True


def test_np_darray(l_val, r_val):
    if l_val.dtype == r_val.dtype and l_val.shape == r_val.shape:
        return (np.nan_to_num(l_val) == np.nan_to_num(r_val)).all();
    return False


def test_dict(l_val, r_val, recursive):
    if l_val.keys() == r_val.keys():
        for key in l_val.keys():
            if not test_equality(l_val[key], r_val[key], recursive):
                return False
        return True
    return False


def copyAndSave(key, value, old_dict, new_dict):
    old_dict[key] = value
    dataCp = copy.deepcopy(value)
    new_dict[key] = dataCp
    return dataCp


def testTwoDic(old_dict: dict, new_dict: dict, function_name: str):
    for name in old_dict.keys():
        if not test_equality(old_dict[name], new_dict[name], True):
            raise Exception(("input[" + name + "] is modified after call function:[$gid[" + function_name + "]]"))


def test_equality(l_val, r_val, recursive):
    if l_val.__class__ == r_val.__class__:
        if isinstance(l_val, pd.DataFrame):
            return test_dataframe(l_val, r_val)
        elif isinstance(l_val, GftTable):
            return test_gft_table(l_val, r_val)
        elif isinstance(l_val, np.ndarray):
            return test_np_darray(l_val, r_val)
        elif isinstance(l_val, np.chararray):
            return test_np_darray(l_val, r_val)
        elif isinstance(l_val, pd.core.indexes.datetimes.DatetimeIndex):
            return test_coord(l_val, r_val)
        elif isinstance(l_val, pd.core.indexes.base.Index):
            return test_coord(l_val, r_val)
        elif isinstance(l_val, dict):
            return test_dict(l_val, r_val, recursive)
        elif l_val.__dir__().__contains__("__dict__"):
            if recursive:
                return test_dict(l_val.__dict__, r_val.__dict__, False)
            elif l_val.__dir__().__contains__("__eq__"):
                return l_val.__eq__(r_val)
            elif l_val.__dir__().__contains__("equals"):
                return l_val.equals(r_val)
            else:
                # blindly return true.
                return True
        else:
            return l_val == r_val
    return False

'''
    for cpp data -> python numpy
'''
def transformInput(data):
    if isinstance(data, int):
        # 62091 is the difference between 1970-01-01 and 1800-01-01
        data = data - 62091
        return pd.to_datetime(data, unit='D')
    if isinstance(data, tuple):
        data = transform2DataFrame(data)
        data = GftTable.fromCppMatrix(data)
        return data
    elif isinstance(data, list):
        columnOrders = [data[i][0] for i in range(len(data))]
        dataDict = transformOTInColumnTable(data)
        data = pd.DataFrame(dataDict)
        data = data[columnOrders]
        data = GftTable.fromColumnTable(data, columnOrders)
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype == np.complex128:
            oArr = transformO4Input(data)
            return pd.Index(data=oArr)
        elif data.dtype == np.int64 or data.dtype == np.uint64:
            tArr = transformTime4Input(data)
            return pd.DatetimeIndex(data=tArr, copy=False)
    elif isinstance(data, dict):
        return transformDict(data)
    return data


def transformDict(dic):
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = transformDict(value)
        else:
            dic[key] = transformInput(value)
    return dic


def gidStrHex2IntPairs(gid_str):
    if gid_str[0].__len__() == 32:
        v1 = int.from_bytes(binascii.a2b_hex(gid_str[0][0:16]), byteorder='big', signed=True)
        v2 = int.from_bytes(binascii.a2b_hex(gid_str[0][16:32]), byteorder='big', signed=True)
        return v1, v2
    elif gid_str == "GFT":
        return int(0), int(0)
    else:
        raise Exception("String [" + gid_str + "] len is not 32")


def printStr(gid_str):
    print(gid_str)
    print(type(gid_str))
    print(gid_str.__len__())


def gidStrArray2CharArray(gid_arr: np.array):
    gid_arr.shape = (gid_arr.shape[0], 1)
    d2 = np.apply_along_axis(gidStrHex2IntPairs, 1, gid_arr)
    # transform int pair two bytes
    return np.chararray(shape=(d2.shape[0]), itemsize=16, buffer=d2.data)


def strSet2Np(gid_arr: np.array):
    return gidStrArray2CharArray(gid_arr)


def testGidStr2Np():
    strArr = np.array(
        ["A6D3CA2532F94917BC5C1DB2F43F5A41", "BDF62F36F82A41A2ADF2C228DB903365", "C6149560360A0DE607CD12AFD003B082"])
    strSet2Np(strArr)


# use this to transform charArray to string
def gidInt2Str(gid):
    if gid.__len__ == 0:
        return "GFT"
    if (gid.__len__() == 16):
        firstO = bytearray(gid[:8])
        firstO.reverse()
        secondO = bytearray(gid[8:])
        secondO.reverse()
        full = firstO.hex() + secondO.hex()
        return full.upper()
    return gid.hex()


def fillNan(row):
    if (np.isnan(row['val'])):
        return gidInt2Str(row['key'])
    else:
        return row['val']


def getDataframe(gft_tab: GftTable):
    gid = gidInt2Str(gid=gft_tab.gid)
    gid = "/home/gft/work/" + gid + ".pkl"
    if gft_tab.matrix is not None:
        with open(gid, "wb", -1) as fps:
            tranfomred_data = transform_bin_o_2_str(gft_tab.matrix)
            fps.write(pickle.dumps(tranfomred_data, -1))
    else:
        with open(gid, "wb", -1) as fps:
            tranfomred_data = transform_bin_o_2_str(gft_tab.columnTab)
            fps.write(pickle.dumps(tranfomred_data, -1))

def loadAndResave(file_name):
    file_name = "/home/gft/data/" + file_name
    with open(file_name, "rb") as fs:
        gft_tab = pickle.loads(fs.read())
        return getDataframe(gft_tab)


def mergeName(ootv, namemap):
    ts = time.time()
    ret = ootv.merge(right=namemap, how='left', left_on=['O_v1', 'O_v2'], right_index=True)
    elapsed = time.time() - ts
    print(elapsed)
    return ret

def rename_o_t_colname_in_place(tb: pd.DataFrame):
    name_list = list()
    for name in tb.columns:
        type = get_column_type(tb, name)
        if type == PARAMETER_TYPE_TIMESTAMP:
            name_list.append("T")
        elif type == PARAMETER_TYPE_UUID:
            name_list.append("O")
        else:
            name_list.append("V")
    tb.columns = name_list
    return tb


def transformOutputWithTypeGiven(obj, par_type):
    if par_type == PARAMETER_TYPE_NUMBER_NUMRIC:
        return float(obj)
    return transformOutput(obj)


def transformMatrixWithGid(panel, gid):
    data = transform2DataFrame(panel)
    data = GftTable.fromCppMatrixWithGid(data, gid)
    return data


def transformMatrix(panel):
    data = transform2DataFrame(panel)
    data = GftTable.fromCppMatrix(data)
    return data


def transformColumnTableWithGid(columnTable, gid):
    columnOrders = [columnTable[i][0] for i in range(len(columnTable))]
    dataDict = transformOTInColumnTable(columnTable)
    data = pd.DataFrame(dataDict)
    data = data[columnOrders]
    data = GftTable.fromColumnTableWithGid(data, gid, columnOrders)
    return data


def transformColumnTable(columnTable):
    columnOrders = [columnTable[i][0] for i in range(len(columnTable))]
    dataDict = transformOTInColumnTable(columnTable)
    data = pd.DataFrame(dataDict)
    data = data[columnOrders]
    data = GftTable.fromColumnTable(data, columnOrders)
    return data



def transformOutput(obj):
    if isinstance(obj, pd.DataFrame):
        return _frameToList(obj)
    elif isinstance(obj, GftTable):
        if obj.matrix is not None:
            return _wideFrameToTuple(obj.matrix)
        elif obj.columnTab is not None:
            return _longFrameToList(obj.columnTab)
        else:
            return None
    elif isinstance(obj, pd.tslib.Timestamp):
        # 62091 is the difference between 1970-01-01 and 1800-01-01
        return int(obj.tz_localize(tz="UTC").timestamp() / 86400) + 62091
    else:
        return obj


def is_serializable_2_cpp(obj):
    invert_op = getattr(obj, "__serialize__", None)
    return callable(invert_op)

def get_pt_name(data, from_meta):
    if not isinstance(data, pd.DataFrame):
        return None

    if data.columns.__contains__(from_meta):
        return from_meta
    for colname in data.columns:
        if get_column_type(data, colname) == PARAMETER_TYPE_TIMESTAMP:
            return colname
    return None


def slice_data_inplace_and_ret(data, pt_name, begin_time: pd.Timestamp, end_time: pd.Timestamp):
    if isinstance(data, GftTable):
        slice_table_inplace(data, begin_time, end_time)
        return data
    elif isinstance(data, pd.DataFrame):
        if ismatrix(data):
            return slice_matrix(data, begin_time, end_time)
        else:
            return slice_column_table(data, pt_name, begin_time, end_time)
    elif isinstance(data, pd.core.indexes.datetimes.DatetimeIndex):  #
        return data[(data > begin_time) & (data <= end_time)]
    return data


def slice_column_table(column_table: pd.DataFrame, pt_name, begin_time: pd.Timestamp, end_time: pd.Timestamp):
    print("Slice col tab to[{0}:{1}]".format(str(begin_time),str(end_time)))
    return column_table.loc[(column_table[pt_name] >= begin_time) & (column_table[pt_name] <= end_time)]


def slice_matrix(matrix: pd.DataFrame, begin_time: pd.Timestamp, end_time: pd.Timestamp):
    print("Slice matrix to[{0}:{1}]".format(str(begin_time), str(end_time)))
    return matrix.loc[(matrix.index >= begin_time) & (matrix.index <= end_time)]


# data['x2'].matrix.index.get_loc('2016-05-29', method='backfill')
# ct[ct['T']<'2015-12-31']['T'].max()


def slice_table_inplace(gft_table: GftTable,pt_name, begin_time: pd.Timestamp, end_time: pd.Timestamp):
    if gft_table.columnTab is not None:
        sliced_col_tables = slice_column_table(gft_table.columnTab, pt_name, begin_time, end_time)
        gft_table.columnTab = sliced_col_tables
        gft_table.matrix = None
    elif gft_table.matrix is not None:
        sliced_matrix = slice_matrix(gft_table.matrix, begin_time, end_time)
        gft_table.matrix = sliced_matrix
        gft_table.columnTab = None
    return gft_table


def wrap_gfttable_dataframe_clean_gid(obj):
    global inst_gid
    ret = wrap_gfttable_dataframe_with_gid(obj, inst_gid)
    inst_gid = None
    return ret


def wrap_gfttable_dataframe_with_gid(obj, gid):
    if isinstance(obj, pd.DataFrame):
        if PARAMETER_TYPE_PANEL == getDataFrameType(obj):
            return GftTable.fromPythonMatrixWithGid(obj, gid)
        else:
            return GftTable.fromColumnTableWithGid(obj, gid, None)
    elif isinstance(obj, GftTable):
        if obj.gid is not None:
            obj.gid = gid
    return obj


def _frameToList(obj: pd.DataFrame):
    if PARAMETER_TYPE_PANEL == getDataFrameType(obj):
        return _wideFrameToTuple(obj)
    else:
        return _longFrameToList(obj)


def _longFrameToList(obj):
    alist = []
    for columnname in obj.columns:
        column_data = obj.loc[:, columnname].values
        if 0 == len(column_data):
            return None
        elif isinstance(column_data[0], np.datetime64):
            column_data = transformTime4Output(column_data)
        elif not isinstance(column_data[0], float):  # O array
            column_data = transformOFromPandas2Output(column_data)

        acolumn = (columnname, column_data)
        alist += [acolumn]
    return alist


def _wideFrameToTuple(obj):
    objT = obj.transpose()
    ls_index = transformOFromPandas2Output(objT.index.values)
    ls_column = transformTime4Output(objT.columns.values)
    arr_values = objT.values
    if (0 == len(ls_index) or 0 == len(ls_column) or 0 == len(arr_values)):
        return None

    return (ls_index, ls_column, arr_values)


class LongTable2Readable:  # long table means OnTnVn, or OOTV
    # Martin: the code below will be executed when load this py file. Need to comment it out
    # uuid2Readable = Uuid2Readable()

    def convert(self, ootv):
        result = []
        for column in ootv:
            arr = column[1]
            name = column[0]
            if len(arr) > 0:
                if (type(arr[0]) == np.complex128):  # O array
                    result.append((name, self.uuid2Readable.oArr2Readable(arr)))
                elif (type(arr[0]) == np.int64):  # T array
                    # please change the code to convert t array
                    result.append(column)
                else:
                    result.append(column)
            else:
                result.append(column)
        return result


def intTuple2Str(tup2Int):
    """
    convert tuple to 32bit hex
    """
    str1 = int.to_bytes(int(tup2Int[0][0]), 8, "big").hex().upper()
    str2 = int.to_bytes(int(tup2Int[0][1]), 8, "big").hex().upper()
    return str1 + str2


def binary_to_str(bs):
    str1 = int.to_bytes(int.from_bytes(bs[:8], byteorder='little'),8,'big').hex().upper()
    str2 = int.to_bytes(int.from_bytes(bs[8:], byteorder='little'), 8, 'big').hex().upper()
    return str1 + str2


def gid_str_to_bin(str):
    b1 = int.to_bytes(int.from_bytes(binascii.a2b_hex(str[:16]), byteorder='big'),8, 'little')
    b2 = int.to_bytes(int.from_bytes(binascii.a2b_hex(str[16:]), byteorder='big'), 8, 'little')
    return b1 + b2


def oSet2Hex(oSet):
    """
    convert array to 32 bit hex array
    """
    ret = oSet.astype(oDataType)
    return np.apply_along_axis(intTuple2Str, 1, ret.reshape(-1, 1))




def get_primary_t_name_with_def(data, pt_name):
    if pt_name is None:
        for name in data.columns:
            if 'datetime64' in str(data[name].dtype):
                return name

    return pt_name



def transform_o_2_name_4_nparray(arr, nameMap, is_binary=True):
    pdArr = pd.DataFrame({'key': arr})
    pdMapped = pdArr.merge(right=nameMap, how='left', left_on='key', right_index=True)

    nullRows = pdMapped.val.isnull()
    if (pdMapped[nullRows].size > 0):
        if is_binary:
            gidName = pdMapped[nullRows].key.apply(gidInt2Str)
        else:
            gidName = pdMapped[nullRows].key
        pdMapped.val[gidName.index] = gidName

    return pdMapped.val

def transform_o_2_name_4_dataframe(df, nameMap, is_binary=True):
    if ismatrix(df):
        return pd.DataFrame(data=df.values, index=df.index,
                            columns=transform_o_2_name_4_nparray(df.columns.values, nameMap, is_binary))
    else:
        # this is column table
        if df.size == 0:
            return None
        rtnDict = dict()
        for col in df.columns:
            if df[col].dtype == np.object:
                rtnDict.__setitem__(col, transform_o_2_name_4_nparray(df[col].values, nameMap, is_binary))
            else:
                rtnDict.__setitem__(col, df[col].values)
        return pd.DataFrame(rtnDict)

def transform_o_2_name(data, namemap, is_binary):
    if (isinstance(data, GftTable)):
        if (data.matrix is None):
            colTable = transform_o_2_name_4_dataframe(data.columnTab, namemap, is_binary)
            return GftTable.fromColumnTable(colTable, data.columnOrders)
        else:
            mat = transform_o_2_name_4_dataframe(data.matrix, namemap, is_binary)
            return GftTable.fromCppMatrix(mat)
    elif isinstance(data, pd.DataFrame):
        return transform_o_2_name_4_dataframe(data, namemap, is_binary)
    elif isinstance(data, dict):
        resultDict = dict()
        for key, value in data.items():
            try:
                resultDict[key] = transform_o_2_name(value, namemap, is_binary)
            except Exception:
                print("transform " + key + ' failed')
                return
        return resultDict
    return data

def transform_bin_o_2_str(data):
    if isinstance(data, dict):
        ret = dict()
        for key, val in data.items():
            ret[key] = transform_bin_o_2_str(val)
        return ret
    if isinstance(data, GftTable):
        if data.columnTab is not None:
            return GftTable.fromColumnTableWithGid(transform_bin_o_2_str(data.columnTab), data.gid, data.columnOrders)
        if data.matrix is not None:
            return GftTable.fromPythonMatrixWithGid(transform_bin_o_2_str(data.matrix), data.gid)
        return data
    if isinstance(data, pd.DataFrame):
        if ismatrix(data):
            ret = data.copy()
            ret.columns = oSet2Hex(ret.columns.values)
            return ret
        else:
            ret = data.copy()
            for name in ret.columns:
                if PARAMETER_TYPE_UUID == get_column_type(ret, name):
                    ret[name] = oSet2Hex(ret[name].values)
            return ret
    return data

def load_transform_and_save_data(data_file_name, file_name, zip_lvl = 5):
    data = zload(data_file_name)
    value = transform_bin_o_2_str(data)
    dps = pickle.dumps(value, -1)
    if zip_lvl > 0:
        zDps = zlib.compress(dps, zip_lvl)
    else:
        zDps = dps
    with open(file_name, "wb", -1) as fpz:
        fpz.write(zDps)

def load_and_save_2_csv_file(data_file, csv_file = None):
    with open(data_file, "rb") as rfs:
        pd_data = pickle.loads(rfs.read())
        if csv_file is not None:
            pd_data.to_csv(csv_file)
        return pd_data


def transform_o_2_name_with_file(dic, map_file_name="lib/gftTools/symbols.pkl"):
    with open(map_file_name, "rb") as fpz:
        value = fpz.read()
        namemap = pickle.loads(value)
    return transform_o_2_name(dic, namemap, True)


def transfrom_str_o_2_name_with_file(dic, map_file_name="lib/gftTools/symbols_str.pkl"):
    with open(map_file_name, "rb") as fpz:
        value = fpz.read()
        namemap = pickle.loads(value)
    return transform_o_2_name(dic, namemap, False)



def slice_column_tab(data, begin_time, end_time, pt_name):
    pt_name = get_primary_t_name_with_def(data, pt_name)
    if pt_name is None:
        return data
    return data.loc[(data[pt_name] >= begin_time) & (data[pt_name] <= end_time)]


import sys

def change_working_dir():
    for path in sys.path:
        if '/home/gft/work' in path:
            import os
            print("change current working dir to: /home/gft/work")
            os.chdir('/home/gft/work')
            return

change_working_dir()


from lib.gftTools import requests_pb2

import urllib.request as urllib2

from lib.gftTools import gsConst


MT_EFU_VIEW = 1
ST_RUN_SKILL_INSTANCE = 25
HTTP_RESP_OK_CODE = 200
req_no = 0

from lib.gftTools import skill_pb2

def test_run_skill_inst(inst_gid, server_url='http://127.0.0.1:9030', user_name = 'GFT', pwd= '123456'):
    # data = extract_info(tf, {'ffmpeg':0})
    req = requests_pb2.RunSkillInstanceRequest()
    req.skilInstanceGid = inst_gid
    req.returnFullNodeDetails = True
    global  req_no
    req_no += 1
    ret = call_vq(req.SerializeToString(), MT_EFU_VIEW, ST_RUN_SKILL_INSTANCE, req_no, server_url, user_name, pwd)
    if ret[0] == HTTP_RESP_OK_CODE:
        ret = skill_pb2.RespRunNodeAction()
        ret.Par


def call_vq(protbuf_data, main_req, sub_req, req_no, server_url='http://127.0.0.1:9030', user_name = 'GFT', pwd= '123456'):
    r = urllib2.Request(server_url + '/vqservice/vq/internalRequest', protbuf_data,
                        {'Content-Type': 'application/octet-stream', gsConst.HTTP_HEADER_PARA_NAME_USER: user_name,
                         gsConst.HTTP_HEADER_PARA_NAME_PASSWORD: pwd, gsConst.HTTP_HEADER_PARA_MR: main_req, gsConst.HTTP_HEADER_PARA_SR: sub_req,
                         gsConst.HTTP_HEADER_PARA_REQ_NO:req_no, gsConst.HTTP_HEADER_PARA_IS_BYTES: 1})
    r.get_method = lambda: 'POST'
    ret = urllib2.urlopen(r)
    ret_code = ret.getcode()
    if ret_code == HTTP_RESP_OK_CODE:
        return ret_code,ret.read()
    return ret, None