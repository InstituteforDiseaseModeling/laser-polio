# *****************************************************************************
#
# *****************************************************************************

import json

# *****************************************************************************


# Rules for name cleaning
def reprule(revval):
    # Upper case
    revval = revval.upper()

    # Diacritics
    revval = revval.replace("Â", "A")
    revval = revval.replace("Á", "A")
    revval = revval.replace("Ç", "C")
    revval = revval.replace("Ê", "E")
    revval = revval.replace("É", "E")
    revval = revval.replace("È", "E")
    revval = revval.replace("Ï", "I")
    revval = revval.replace("Ã¯", "I")
    revval = revval.replace("Í", "I")
    revval = revval.replace("Ñ", "NY")
    revval = revval.replace("Ô", "O")
    revval = revval.replace("Ó", "O")
    revval = revval.replace("Ü", "U")
    revval = revval.replace("Û", "U")
    revval = revval.replace("Ú", "U")

    # Alias characters to underscore
    revval = revval.replace(" ", "_")
    revval = revval.replace("-", "_")
    revval = revval.replace("/", "_")
    revval = revval.replace(",", "_")
    revval = revval.replace("\\", "_")

    # Remove ASCII characters
    revval = revval.replace("'", "")
    revval = revval.replace('"', "")
    revval = revval.replace("’", "")
    revval = revval.replace(".", "")
    revval = revval.replace("(", "")
    revval = revval.replace(")", "")
    revval = revval.replace("\x00", "")

    # Remove non-ASCII characters
    revval = revval.encode("ascii", "replace")
    revval = revval.decode()
    revval = revval.replace("?", "")

    # Condence and strip underscore characters
    while revval.count("__"):
        revval = revval.replace("__", "_")
    revval = revval.strip("_")

    return revval


# *****************************************************************************


def make_cbr_dict():
    with open("data00.json") as fid01:
        dict_adm00 = json.load(fid01)

    with open("data01.json") as fid01:
        dict_adm01 = json.load(fid01)

    with open("NGA_NAMES_LEV02.csv") as fid01:
        nga_names = [val.strip().split(":") for val in fid01.readlines()]
    nga_lev01 = list({":".join(val[:-1]) for val in nga_names})
    list({":".join(val) for val in nga_names})

    res_dict = {}
    ref_val = dict_adm00["Data"][0]["Value"]

    for reg_dict in dict_adm01["Data"]:
        reg_name = reg_dict["CharacteristicLabel"]
        reg_val = reg_dict["Value"]
        if ".." not in reg_name:
            continue
        dotname = "AFRO:NIGERIA:" + reprule(reg_name[2:])
        if dotname not in nga_lev01:
            raise ValueError(f"{dotname} not in {nga_lev01}")  # 1/0
        res_dict[dotname] = reg_val / ref_val

    with open("cbr_NGA.json", "w") as fid01:
        json.dump(res_dict, fid01, indent=4, sort_keys=True)

    return None


# ******************************************************************************


if __name__ == "__main__":
    make_cbr_dict()

# ******************************************************************************
