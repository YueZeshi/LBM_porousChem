from ruamel.yaml import YAML
yaml = YAML()
file = "data/gri30.yaml"
# print(gas.report())
with open(file,"r",encoding="utf-8")as f:
    data = yaml.load(f)
# ['description', 'generator', 'input-files', 'cantera-version', 'date', 'units', 'phases', 'species', 'reactions']
# units 存单位 {length: cm, time: s, quantity: mol, activation-energy: cal/mol}
## length time quantity activation-energy
# phases: name themo elements species kinetics transport state
# species list[name composition thermo[model temperature-ranges分段 data分段 note注释] transport[model geometry well-depth diameter polarizability rotational-relaxtion]]
# reactions list[equation+] default->rate-constant (duplicate); type:three-body + efficiencies; type:falloff+ low-P-rate-constant+high-P-rate-constant+ Troe+ efficiencies
# 