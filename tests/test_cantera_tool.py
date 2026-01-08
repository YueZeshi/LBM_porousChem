

# test_cantera_tool.py
import pytest
from LBM.cantera_tool import ck2yaml

def test_flowmap_and_flowlist():
    m = ck2yaml.FlowMap(a=1, b=2)
    assert isinstance(m, dict)
    l = ck2yaml.FlowList([1, 2, 3])
    assert isinstance(l, list)
    assert l == [1, 2, 3]

def test_strip_nonascii():
    s = "abc中文Ω"
    result = ck2yaml.strip_nonascii(s)
    assert all(ord(c) < 128 for c in result)

def test_error_formatter():
    import logging
    formatter = ck2yaml.ErrorFormatter()
    record = logging.LogRecord("test", logging.ERROR, "", 0, "error!", (), None)
    msg = formatter.format(record)
    assert msg.startswith("*")

def test_input_error():
    with pytest.raises(ck2yaml.InputError):
        raise ck2yaml.InputError("test error")

def test_species_str_and_yaml():
    sp = ck2yaml.Species("H2O")
    assert "H2O" in str(sp)
    # to_yaml 需配合 ruamel.yaml 使用，略

# 可根据实际需求补充对 Reaction、Nasa7、Arrhenius 等对象的构造与属性测试