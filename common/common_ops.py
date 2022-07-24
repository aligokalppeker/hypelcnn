import ntpath


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def replace_abbrs(txt, abbrs_dict):
    for word, abbr in abbrs_dict.items():
        txt = txt.replace(word, abbr)
    return txt


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
