library_list = [
    'esoctiostan', 'esohststan', 'esookestan', 'esowdstan', 'esoxshooter',
    'ing_oke', 'ing_sto', 'ing_og', 'ing_mas', 'ing_fg', 'irafblackbody',
    'irafbstdscal', 'irafctiocal', 'irafctionewcal', 'irafiidscal',
    'irafirscal', 'irafoke1990', 'irafredcal', 'irafspec16cal',
    'irafspec50cal', 'irafspechayescal'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/hamuystandards.html
esoctiostan = [
    'cd32d9927', 'cd_34d241', 'eg21', 'eg274', 'feige110', 'feige56',
    'hilt600', 'hr1544', 'hr3454', 'hr4468', 'hr4963', 'hr5501', 'hr718',
    'hr7596', 'hr7950', 'hr8634', 'hr9087', 'ltt1020', 'ltt1788', 'ltt2415',
    'ltt3218', 'ltt3864', 'ltt4364', 'ltt4816', 'ltt6248', 'ltt7379', 'ltt745',
    'ltt7987', 'ltt9239', 'ltt9491'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/hststandards.html
esohststan = [
    'agk81d266', 'bd28d4211', 'bd33d2642', 'bd75d325', 'bpm16274', 'feige110',
    'feige34', 'g191b2b', 'g93_48', 'gd108', 'gd50', 'grw70d5824', 'hd49798',
    'hd60753', 'hd93521', 'hr153', 'hr1996', 'hr4554', 'hr5191', 'hr7001',
    'hz2', 'hz21', 'hz4', 'hz44', 'lb227', 'lds749b', 'ngc7293'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/okestandards_rev.html
esookestan = [
    'bd25d4655', 'bd28d4211', 'bd33d2642', 'bd75d325', 'feige110', 'feige34',
    'feige66', 'feige67', 'g138_31', 'g158_100', 'g191b2b', 'g193_74', 'g24_9',
    'g60_54', 'gd108', 'gd248', 'gd50', 'grw70d5824', 'hd93521', 'hz21', 'hz4',
    'hz44', 'ltt9491', 'ngc7293', 'sa95_42'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/wdstandards.html
esowdstan = [
    'agk_81d266_005', 'alpha_lyr_004', 'bd_25d4655_002', 'bd_28d4211_005',
    'bd_33d2642_004', 'bd_75d325_005', 'feige110_005', 'feige34_005',
    'feige66_002', 'feige67_002', 'g93_48_004', 'gd108_005', 'gd50_004',
    'gd71', 'grw_70d5824_005', 'hd93521_005', 'hz21_005', 'hz2_005',
    'hz44_005', 'hz4_004', 'lb227_004', 'lds749b_005', 'ltt9491_002',
    'ngc7293_005'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/Xshooterspec.html
esoxshooter = ['EG274', 'Feige110', 'GD153', 'GD71', 'LTT3218', 'LTT7987']

# http://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/workflux.html
ing_oke = [
    'bd254', 'bd28', 'bd33', 'bd75', 'erib', 'f110', 'f24', 'f34', 'f66',
    'f67', 'g138', 'g158', 'g191new', 'g191old', 'g193', 'g24', 'g47', 'g60',
    'g99', 'gd108', 'gd140', 'gd190', 'gd248', 'gd50', 'grw705new',
    'grw705old', 'grw708', 'grw73', 'hd935', 'he3', 'hz14', 'hz2', 'hz21',
    'hz29', 'hz43', 'hz44new', 'hz44old', 'hz4new', 'hz4old', 'hz7', 'l1363',
    'l1512', 'l745', 'l870', 'l930', 'l970', 'lb1240', 'lb227', 'lds235',
    'lds749', 'ltt', 'ngc', 'r627', 'r640', 'sa29', 'sa95', 't573', 'w1346',
    'w485'
]

ing_sto = [
    'bd08', 'bd253', 'bd28', 'bd33', 'bd40', 'f110', 'f15', 'f25', 'f34',
    'f56', 'f92', 'f98', 'h102', 'h600', 'hz15', 'k27'
]

ing_og = ['bd17', 'bd26', 'hd194', 'hd849']

ing_mas = [
    'bd28', 'cyg', 'eg81', 'f110', 'f34', 'f66', 'f67', 'g191', 'gd140',
    'h600', 'hd192', 'hd217', 'hz14', 'hz44', 'pg0205', 'pg0216', 'pg0310',
    'pg0823', 'pg0846', 'pg0934', 'pg0939', 'pg1121', 'pg1545', 'pg1708',
    'w1346'
]

ing_fg = ['g138', 'g158', 'g24', 'gd248']

# The following iraf standards refer to:
# https://github.com/iraf-community/iraf/tree/master/noao/lib/onedstds
irafblackbody = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'L', 'Lprime', 'M']

irafbstdscal = [
    'hr718', 'hr3454', 'hr3982', 'hr4468', 'hr4534', 'hr5191', 'hr5511',
    'hr7001', 'hr7596', 'hr7950', 'hr8634', 'hr9087', 'hr15318', 'hr74280',
    'hr100889', 'hr188350', 'hr198001', 'hr214923', 'hr224926'
]

irafctiocal = [
    'bd8', 'bd25', 'bd73632', 'cd32', 'eg11', 'eg21', 'eg26', 'eg31', 'eg54',
    'eg63', 'eg76', 'eg79', 'eg99', 'eg139', 'eg149', 'eg158', 'eg248',
    'eg274', 'f15', 'f25', 'f56', 'f98', 'f110', 'feige15', 'feige25',
    'feige56', 'feige98', 'feige110', 'g2631', 'g9937', 'g16350', 'h600',
    'hz2', 'hz4', 'hz15', 'kopf27', 'l377', 'l1020', 'l1788', 'l2415', 'l2511',
    'l3218', 'l3864', 'l4364', 'l4816', 'l6248', 'l7379', 'l7987', 'l8702',
    'l9239', 'l9491', 'l74546', 'l93080', 'l97030', 'lds235', 'lds749',
    'ltt4099', 'ltt8702', 'rose627', 'w1346', 'w485a', 'wolf1346', 'wolf485a'
]

irafctionewcal = [
    'cd32', 'eg21', 'eg274', 'f56', 'f110', 'h600', 'l377', 'l745', 'l1020',
    'l1788', 'l2415', 'l2511', 'l3218', 'l3864', 'l4364', 'l4816', 'l6248',
    'l7379', 'l7987', 'l9239', 'l9491', 'cd32blue', 'eg21blue', 'eg274blue',
    'f56blue', 'f110blue', 'h600blue', 'l377blue', 'l1020blue', 'l1788blue',
    'l2415blue', 'l2511blue', 'l3218blue', 'l3864blue', 'l4364blue',
    'l4816blue', 'l6248blue', 'l7379blue', 'l7987blue', 'l9239blue',
    'l9491blue', 'cd32red', 'eg21red', 'eg274red', 'f56red', 'f110red',
    'h600red', 'l377red', 'l745red', 'l1020red', 'l1788red', 'l2415red',
    'l2511red', 'l3218red', 'l3864red', 'l4364red', 'l4816red', 'l6248red',
    'l7379red', 'l7987red', 'l9239red', 'l9491red'
]

irafiidscal = [
    '40erib', 'amcvn', 'bd7781', 'bd73632', 'bd82015', 'bd253941', 'bd284211',
    'bd332642', 'bd404032', 'eg11', 'eg20', 'eg26', 'eg28', 'eg29', 'eg31',
    'eg33', 'eg39', 'eg42', 'eg50', 'eg54', 'eg63', 'eg67', 'eg71', 'eg76',
    'eg77', 'eg79', 'eg91', 'eg98', 'eg99', 'eg102', 'eg119', 'eg129', 'eg139',
    'eg144', 'eg145', 'eg148', 'eg149', 'eg158', 'eg162', 'eg182', 'eg184',
    'eg193', 'eg247', 'eg248', 'feige15', 'feige24', 'feige25', 'feige34',
    'feige56', 'feige92', 'feige98', 'feige110', 'g88', 'g2610', 'g2631',
    'g4718', 'g9937', 'g12627', 'g14563', 'g16350', 'g191b2b', 'gd128',
    'gd140', 'gd190', 'gh7112', 'grw705824', 'grw708247', 'grw738031', 'he3',
    'hz2', 'hz4', 'hz7', 'hz14', 'hz15', 'hz29', 'hz43', 'hz44', 'kopff27',
    'hiltner102', 'hiltner600', 'l8702', 'l13633', 'l14094', 'l74546a',
    'l93080', 'l97030', 'l140349', 'l151234b', 'lft1655', 'lb227', 'lb1240',
    'lds235b', 'lds749b', 'lp414101', 'ltt4099', 'ltt8702', 'ltt13002',
    'ltt16294', 'ross627', 'ross640', 'sa29130', 'sao131065', 'ton573',
    'wolf1346', 'wolf485a'
]

irafirscal = [
    'bd082015', 'bd174708', 'bd253941', 'bd262606', 'bd284211', 'bd332642',
    'bd404032', 'eg50', 'eg71', 'eg139', 'eg158', 'eg247', 'feige15',
    'feige25', 'feige34', 'feige56', 'feige92', 'feige98', 'feige110',
    'g191b2b', 'hd2857', 'hd17520', 'hd19445', 'hd60778', 'hd74721', 'hd84937',
    'hd86986', 'hd109995', 'hd117880', 'hd161817', 'hd192281', 'hd217086',
    'he3', 'hiltner102', 'hiltner600', 'hr7001', 'hz44', 'kopff27', 'wolf1346'
]

irafoke1990 = [
    'bd75325', 'bd284211', 'feige34', 'feige67', 'feige110', 'g249', 'g13831',
    'g191b2b', 'g19374', 'gd108', 'gd248', 'hz21', 'ltt9491', 'eg71', 'eg158',
    'eg247'
]

irafredcal = [
    '40erib', 'amcvn', 'bd7781', 'bd73632', 'bd174708', 'bd262606', 'eg20',
    'eg33', 'eg50', 'eg54', 'eg63', 'eg67', 'eg76', 'eg79', 'eg91', 'eg98',
    'eg99', 'eg102', 'eg119', 'eg129', 'eg139', 'eg144', 'eg145', 'eg148',
    'eg149', 'eg158', 'eg162', 'eg182', 'eg184', 'eg193', 'eg247', 'eg248',
    'feige24', 'g2610', 'g2631', 'g4718', 'g9937', 'g12627', 'g14563',
    'g16350', 'g191b2b', 'gd140', 'gd190', 'grw705824', 'grw708247',
    'grw738031', 'hd19445', 'hd84937', 'he3', 'hz29', 'hz43', 'hz44', 'l13633',
    'l14094', 'l151234b', 'l74546a', 'l93080', 'l97030', 'lds235b', 'lds749b',
    'lft1655', 'ltt4099', 'ltt8702', 'ltt16294', 'ross627', 'ross640',
    'sa29130', 'sao131065', 'wolf1346', 'wolf485a'
]

irafspec16cal = [
    'hd15318', 'hd30739', 'hd74280', 'hd100889', 'hd114330', 'hd129956',
    'hd188350', 'hd198001', 'hd214923', 'hd224926', 'hr718', 'hr1544',
    'hr3454', 'hr4468', 'hr4963', 'hr5501', 'hr7596', 'hr7950', 'hr8634',
    'hr9087', 'hd15318blue', 'hd30739blue', 'hd74280blue', 'hd100889blue',
    'hd114330blue', 'hd129956blue', 'hd188350blue', 'hd198001blue',
    'hd214923blue', 'hd224926blue', 'hr718blue', 'hr1544blue', 'hr3454blue',
    'hr4468blue', 'hr4963blue', 'hr5501blue', 'hr7596blue', 'hr7950blue',
    'hr8634blue', 'hr9087blue', 'hd15318red', 'hd30739red', 'hd74280red',
    'hd100889red', 'hd114330red', 'hd129956red', 'hd188350red', 'hd198001red',
    'hd214923red', 'hd224926red', 'hr718red', 'hr1544red', 'hr3454red',
    'hr4468red', 'hr4963red', 'hr5501red', 'hr7596red', 'hr7950red',
    'hr8634red', 'hr9087red'
]

irafspec50cal = [
    'bd284211', 'cygob2no9', 'eg20', 'eg42', 'eg71', 'eg81', 'eg139', 'eg158',
    'eg247', 'feige34', 'feige66', 'feige67', 'feige110', 'g191b2b', 'gd140',
    'hd192281', 'hd217086', 'hilt600', 'hz14', 'hz44', 'pg0205134',
    'pg0216032', 'pg0310149', 'pg0823546', 'pg0846249', 'pg0934554',
    'pg0939262', 'pg1121145', 'pg1545035', 'pg1708602', 'wolf1346'
]

irafspechayescal = [
    'bd284211', 'cygob2no9', 'eg42', 'eg71', 'eg81', 'eg139', 'eg158', 'eg247',
    'feige34', 'feige66', 'feige67', 'feige110', 'g191b2b', 'gd140',
    'hd192281', 'hd217086', 'hilt600', 'hz14', 'hz44', 'pg0205134',
    'pg0216032', 'pg0310149', 'pg0823546', 'pg0846249', 'pg0934554',
    'pg0939262', 'pg1121145', 'pg1545035', 'pg1708602', 'wolf1346'
]
