# em_eqf 地震目录来源与处理

## 诱发地震观测目录与作业记录

每个数据集的 `data/<名称>/raw/` 保存发布方原件，`sources.json` 记录来源、署名、许可及 SHA-256。`processed/` 中的 `*_eq_processed.csv` 为逐事件表，`*_inj_processed.csv` 为配套注入观测，`*_summary.json` 记录筛选及缺测数。时间、震级、注入量按各来源的实际分辨率保留。运行 `python scripts/data/prepare_induced_catalogs.py` 可重新校验原件并生成处理表。

| 数据集 | 已入库事件（有震级） | 首末事件年份 | 可使用的作业记录 |
|---|---:|---|---|
| `Paradox_Valley`，美国科罗拉多州 | 5,341（5,341） | 1991–2013 | 6,815 条日均注入流量和地面压力记录 |
| `Newberry_EGS2012`，美国俄勒冈州 | 172（172） | 2012–2013 | 89,212 条井下流量与井口压力记录，通常每分钟一次 |
| `Pohang_EGS`，韩国浦项 | 98（98） | 2015–2017 | 两井五轮刺激的 16,747 条流量和井口压力记录 |
| `Aspo_HRL2015`，瑞典 Äspö 地下实验室 | 19,712（4,287） | 2015 | 六次水力压裂、七段记录，按秒汇总流量和区间压力 |
| `KNMI_Induced`，荷兰及周边台网覆盖区 | 4,316（4,316） | 1986–2025 | 官方诱发或触发事件目录 |

- **Paradox Valley。** 美国垦务局原目录含 6,033 条。处理表采用注入年代的 a 级相对定位事件，按报告限定深度小于井口以下 8.5 km（井口海拔 1,540 m），保留 MD 和 ML 标签。日均流量按美制加仑换算，时间列保留来源日历日期，1 个缺流量日留空。震例的归因适用于注入区序列，位置等级表示定位方法。[原件与技术报告](https://www.usbr.gov/uc/progact/paradox/index.html) · [井口高程依据](https://www.usbr.gov/uc/DocLibrary/Reports/Paradox/uc-2024annualreport-paradoxvalleyseismicnetwork-508-20250800.pdf)
- **Newberry。** 采用作者重定位版本；震级类型原字段为下划线，在处理表中留空。两条同秒事件的坐标和震级均不同，保留两条。流量取通往井下的通道，保留仪器已校正的数值。按记录顺序解析 2012 年秋季太平洋时区夏令时回拨，再转 UTC。两个状态文本对应的流量留空。泵表止于 2012 年 12 月，之后仍有震例。[GDR 井 55-29 原始数据](https://doi.org/10.15121/1357916)
- **浦项。** 地震取论文图 2a 的事件表，保留 98 个 ML 和其中 46 个 Mw。这是论文选用的事件集，完整性分析需另行选择目录。作业记录只取图 2b–f 的实测表，按井号和刺激编号区分，KST 转 UTC，L/s 转 m³/min。停测期保持缺测。[作者数据工作簿及论文](https://doi.org/10.1038/s41467-020-16408-0)
- **Äspö。** 连续声发射目录共 19,756 行。按 GFZ 字段说明从事件名读取 UTC；另一列的秒位与日期中的日相等，全表已逐行复核，其原值保留在处理表中。去掉 11 条完全重复行，剔除同一标识下相互冲突的 2 条记录，再选取水力记录覆盖的事件。缺相对声发射震级 MAE 的 15,425 个事件保留供时间分析。局部坐标为 Äspö96。每条流量仅汇总截至该秒的观测，记录各秒的采样数；发布方给出的水力记录绝对授时精度为 1 秒。[GFZ 数据](https://doi.org/10.5880/GFZ.2.6.2023.004) · [采集与实验说明](https://essd.copernicus.org/articles/16/295/2024/)
- **KNMI。** QuakeML 查询范围截至 2026-01-01，事件类型限定为 `induced or triggered event`。4,317 条返回记录中，采用人工定位且状态为 final、reviewed 或 confirmed 的 4,316 条。保留机构事件 ID、首选震源和首选震级 ID，以及 MLn、MLnq、MLs 的区别。覆盖 KNMI 服务区的多个地点。[KNMI FDSN 服务](https://rdsa.knmi.nl/fdsnws/event/1/)

## EGS Collab 诱发微震（美国 SURF，2018 年 5 月）

这是南达科他州同一地下实验场 E1-I 井 164 ft 段的五次水力刺激。场地计数为 1。地震来自 Schoenball 等的更新人工目录 v200629，泵注来自 Knox 等的原始记录。两份数据均采用 CC BY 4.0。震级字段是作者的相对震级 `mag_rel`，位置是矿区局部坐标。原始全目录共 1,932 条，两个缺失震级保留在原表中。[微震数据](https://doi.org/10.15121/1557417) · [泵注记录](https://doi.org/10.15121/1651116) · [试验时序和流量核对](https://doi.org/10.1029/2020JB020840)

| 数据集 | 联合观测窗内事件 | 默认 Mc | 训练 / 验证 / 测试计分事件 |
|---|---:|---:|---:|
| EGSCollab_522 | 38 | −4.25 | 16 / 4 / 5 |
| EGSCollab_523 | 138 | −4.45 | 75 / 17 / 19 |
| EGSCollab_524 | 298 | −4.25 | 163 / 38 / 35 |
| EGSCollab_525A | 75 | −4.25 | 32 / 9 / 9 |
| EGSCollab_525B | 113 | −4.35 | 68 / 17 / 16 |

每次刺激单独建序列。观测边界取公开波形文件时间与实测流量的交集，向内取整分钟。波形文件通常约每 34 秒一个，相邻文件起点间隔超过 50 秒即分段。524 取含主刺激的第二段，第一段是刺激前的单独记录。525B 的流量先于波形结束，仅使用共同覆盖段。这五段适合比较刺激期的条件预测；研究全期背景率时需要另建包含空段的观测窗。

按 UTC 排序后，在震级筛选前锁定事件数 70% 和 85% 处相邻事件的时间中点。各段仅用训练部分估计 MAXC（0.1 宽震级箱），沿用诱发目录严格大于 Mc 的筛选。小样本下该阈值是探索性估计。验证和测试可读取本段内此前的历史。做跨场地测试时五段应共同留出；同场地跨阶段试验也需在源训练段确定阈值，重新声明目标数据的使用范围。

522、523 使用 Quizix Flow，524、525A、525B 使用 Triplex Net Flow。分钟 t 的流量仅聚合 (t−60 秒, t] 内的观测，单位从 L/min 转成 m³/min；保留带符号读数与每分钟覆盖秒数。默认向模型提供物理流量，启用归一化后仅用训练时段拟合。局部 x/y/z 与定位质量字段保留在事件表。地理经纬度与海拔基准深度没有从局部坐标推算。

共享原件、下载链接、作者和 SHA-256 见 `data/EGSCollab_E1/sources.json` 及同目录的 `raw/`。泵注压缩包保留本次用到的三个完整原始 CSV 和仪器字典。运行 `python scripts/data/prepare_egs_collab.py` 可校验输入并重建五套 `data/EGSCollab_*/processed/`。训练配置使用表中的数据集名，例如 `dataset=EGSCollab_524`。现有多区域 benchmark 的训练清单独立维护。

核查日期 2026-11-06。本文汇总项目扩充的区域目录、EarthquakeNPP 基准、历史实验输入，以及台湾、东安纳托利亚、智利与希腊研究目录。

远端项目在 `/root/autodl-tmp/em_eqf`。原始快照和较大的缓存存于 `/root/autodl-fs/em_eqf_catalogs`，项目通过 `data/<dataset>/` 下的软链接访问。下表记录数对应各批独立标准缓存。多区域实验实际使用的事件以其 `data/MultiRegion/catalogs/expanded_20261106/dataset.json` 为准。

## 目录完整性审计

2026-11-06 已复核 34 个目录条目和 27 个训练包配置，重点检查字段缺失、时间断档及震级完备性。审计报告含 NCEDC 的类型字段回补候选，以及 Montana、台湾 CWA、Chile Potin 等目录的空窗来源。远端报告位于 `experiments/revision/catalog_integrity_20261106/reports/`。[审计报告](experiments/revision/catalog_integrity_20261106/reports/audit_report_zh.md)

修复已另建版本。NCEDC 回补通过原质量门槛的 550 条记录，Montana 全时段统一用 ComCat 已复核地震，CWA 与 Potin 缩短早期观测窗口。另有 EAFZ 和 Santorini 的训练侧震级门槛对照及逐目录字段可用表。远端在 `experiments/revision/catalog_remediation_20261106/reports/`。[修复与复核报告](experiments/revision/catalog_remediation_20261106/reports/remediation_report_zh.md)

## 可直接下载的平台

| 平台 | 可以取得的数据与本次核查 |
|---|---|
| ISC Seismological Dataset Repository | 按地区、关键词或 DOI 找作者提交的事件目录，条目通常附文件、字段说明、论文和许可。西南挪威、Bitdalsvatn 已用该平台的 ZIP 建库，本轮西南挪威直链返回 HTTP 200。另有南部东非裂谷、死海断裂带和 Samos 余震目录可继续检查。[目录检索](https://www.isc.ac.uk/dataset_repository/search.php) · [西南挪威直链](https://download.isc.ac.uk/datarepository/marck_1.zip) |
| GFZ Data Services | 按 DOI 下载地学研究数据和字段说明。此前已获取 IPOC-Deep。另有 Sippl 等的北智利 2007–2021 目录，论文报告 182,847 条并标注采矿相关事件，本轮远端访问该条目下载页超时。[IPOC-Deep](https://dataservices.gfz.de/10.5880/fidgeo.2026.050/ipoc-deep-a-new-seismicity-catalog-for-northern) · [Sippl 数据条目](https://doi.org/10.5880/GFZ.4.1.2023.004) · [配套论文](https://doi.org/10.1016/j.jsames.2023.104326) |
| Caltech SCEDC 替代目录库 | 集中发布南加州 QTM、Hauksson 波形重定位及特定序列双差结果。1981–2025 Hauksson 文件的直链已实测 HTTP 200。它和 SCSN、QTM、White 的区域与年代有交集，使用时按留出时间及同震审计管理。[目录列表](https://scedc.caltech.edu/eq-catalogs/altcatalogs.html) · [Hauksson 数据文件](https://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_DD/sc_1981_2025_1d_3d_gc_soda_noqb_10_1_SCSN.gc) |
| Zenodo | 在论文 Data Availability 查 DOI，进入条目后选择事件表并锁定版本。本轮从版本 `15910095` 下载 `events_catalog.csv`，验证 Zenodo 公布的 MD5，许可证为 CC BY 4.0。[东安纳托利亚版本](https://doi.org/10.5281/zenodo.15910095) · [事件表直链](https://zenodo.org/api/records/15910095/files/events_catalog.csv/content) |
| CWA 公开资料平台与 BSSA 补充材料 | 台湾官方历史年包与 Wu 等重定位 ZIP 均可直接下载。GDMS 短窗查询已验证返回小震，一年窗口返回登录要求。年度批量数据可用下表直链。[CWA 官方产品说明](https://www.cwa.gov.tw/Data/data_catalog/5-1-3.pdf) · [GDMS 下载页](https://gdms.cwa.gov.tw/catalogDownload.php) · [BSSA 作者附件](https://seismosoc.org/Publications/BSSA_html/bssa_98-3/2007166-esupp/) |

全球及区域查询可用 ISC Reviewed Bulletin、USGS ComCat/FDSN 和 EPOS 的 EMSC 参数服务。ISC Reviewed Bulletin 另提供逐月 ISF 文件，ISC-GEM 主目录提供整体 CSV，适合全球中强震用途。它们汇集多个观测网，跨源合并需要做事件匹配。[ISC Reviewed Bulletin](https://isc.ac.uk/iscbulletin/) · [ISC-GEM 下载](https://www.isc.ac.uk/iscgem/download.php) · [USGS FDSN](https://earthquake.usgs.gov/fdsnws/event/1/) · [EPOS 服务](https://www.epos-eu.org/tcs/seismology/data-services)

## 台湾与东安纳托利亚

| 标准目录 | 原始记录 | 标准记录 | 训练 / 验证 / 测试 | 数据下载 |
|---|---:|---:|---:|---|
| `Taiwan_CWA-Standard` | 73,729 | 40,693 | 24,942 / 4,897 / 10,854 | CWA 历史年包含 1990–2025 年 36 个 XML。[ZIP](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Earthquake/E-A0073-002.zip) |
| `Taiwan_Wu2008-Standard` | 267,211 | 33,011 | 24,546 / 4,533 / 3,932 | Wu 等发表的 1991–2005 年重定位年表，按实际 ZIP 逐年计数。[ZIP](https://seismosoc.org/Publications/BSSA_html/bssa_98-3/2007166-esupp/catalogs_1991-2005.zip) |
| `Turkiye_EAFZ-Standard` | 9,442 | 8,220 | 1,214 / 319 / 6,687 | Colavitti 等的 CASP/NLLoc 重定位研究目录。[固定版事件表](https://zenodo.org/api/records/15910095/files/events_catalog.csv/content) · [论文](https://essd.copernicus.org/articles/17/3089/2025/) |

- **台湾 CWA。** 逐年核对 XML 声明条数和 UTC 年界。将每条记录显式的 `+08:00` 转成 UTC。使用 1991–2025 年、ML≥3、源复核码 F、至少 8 台站、gap≤180°、RMS≤1 s、水平误差≤2 km、深度误差≤5 km。保留源质量等级与复核码。验证起点 2019-01-01，测试起点 2022-01-01。
- **台湾 Wu。** 通过集集和 2002 年花莲外海主震及 CWA 重叠记录核对源时钟，按 UTC 解析到百分之一秒。保留作者震级，使用 M≥2.5、A/B、自由深度、至少 8 台站以及与 CWA 相同的 gap、RMS 和误差门槛。验证起点 2002-01-01，测试起点 2004-01-01。
- **东安纳托利亚。** 与论文图 12 对照核对 UTC。使用重新标定的 ML≥2.5、至少 12 个 P/S 震相、gap≤180°、水平误差≤2 km、深度误差≤5 km。震相数和台站数分别解释。作者先按地面运动数据质量和震级范围选样，2023 年两次主震未纳入这份研究事件表。对时间、经纬度、深度和震级均完全相同的 8 组记录，按较小 AFAD ID 保留代表记录，全部源行及别名表另存。验证起点 2022-01-01，测试起点 2023-01-01，观测终点 2024-03-01。
- **同震与实验使用。** 在 2 秒、15 km、1 个震级单位的窗口内，两套台湾目录有 10,805 对候选。CWA、Wu 与旧 `ENPP_JapanDeprecated` 分别有 316、80 对候选。两套台湾目录适合作为同一区域的替代来源。东安纳托利亚从 2019 年开始，可独立做区域研究。现有多区实验的源训练在 2018-07-01 截止，源验证在 2019-01-01 截止。
- **验收。** 三套标准缓存均已逐事件对照物理字段与 UTC，验证 float64 间隔、三个计分集合和实际 DataLoader。旧窗口切分器会提示起始上下文或历史不足窗口，实际计分间隔检查通过。171 份既有保护文件哈希复核通过。

原始文件在 `/root/autodl-fs/em_eqf_catalogs/source_expansion_20261106/raw/`，名称分别为 `cwa_historical.zip`、`taiwan_wu_1991_2005.zip`、`eafz_events_catalog.csv`。Zenodo 元数据另存 `eafz_zenodo.json`。各目录的 `processed/normalized_all.parquet` 保存解析记录，`processed/quality_screened.parquet` 用于标准缓存。批次报告在 `experiments/revision/catalog_source_expansion_20261106/reports/`，包括 `sources.json`、`summary.json`、`overlap_candidates.csv`、`eafz_identical_physical_records.csv`。

## 智利与希腊 DOI 目录

| 标准目录 | 发布文件记录 | 标准记录 | 训练 / 验证 / 测试 | UTC 观测区间 |
|---|---:|---:|---:|---|
| `Chile_Potin-Standard` | 118,004 | 45,143 | 23,455 / 10,840 / 10,848 | `[2000-01-01, 2020-07-01)` |
| `Chile_Atacama-Standard` | 166,403 | 17,027 | 9,195 / 3,551 / 4,281 | `[2020-11-01, 2024-02-15)` |
| `Greece_WestHellenic-Standard` | 21,331 | 2,866 | 1,107 / 926 / 833 | `[2023-12-01, 2024-10-01)` |
| `Greece_SantoriniSwarm-Standard` | 25,484 | 13,036 | 6,447 / 4,673 / 1,916 | `[2025-02-05, 2025-03-01)` |

- **智利全国 Potin。** 下载作者发表版 ZIP，读取主事件表。保留原始震级类型码，清除空格后取本地震级 ML≥3、重定位 RMS≤0.5 s。用 Maule 与 Iquique 主震支持 UTC 解析。Illapel 主震在该文件中未匹配到。验证与测试起点为 2014-01-01、2017-01-01。[作者档案](https://zenodo.org/records/13146436)
- **智利阿塔卡马。** 本机代理从 Zenodo 下载完整 `catalog_temp.csv`，上传后按发布 MD5 复核。保留作者的四个自然事件分类及 GrowClust 重定位解，筛掉矿爆、气枪和疑似错定位。另要求至少 12 个震相、至少 5 个台站同时有 P 和 S、gap≤180°、水平和垂直不确定度分别≤2 和 5 km、ML≥1.5。验证与测试起点为 2022-07-01、2023-04-01。[作者档案](https://zenodo.org/records/15083298) · [目录方法](https://doi.org/10.1029/2025JB031256)
- **西希腊。** 使用作者最终 `catalog_NLL_SSST_coh.csv`。原文件两个椭圆轴列同名，逐行验证第一列较大，并用论文均值交叉确认。要求至少 16 个震相观测、gap≤240°、RMS≤0.35 s、较大椭圆轴及垂直误差≤5 km，固定 ML≥1.3。验证与测试起点为 2024-04-01、2024-07-01。[作者档案](https://zenodo.org/records/18833544) · [配套论文](https://doi.org/10.1029/2026GL122817)
- **Santorini–Amorgos。** 使用作者筛过的重定位 CSV，34 条时空锚点与 USGS UTC 一致，最大时差约 1.65 秒。只在 2 月 5–28 日密集震群段计分，要求 Mamp≥1.5、至少 12 个震相、gap≤180°、RMS≤0.35 s，水平与垂直不确定度分别≤2 和 5 km。验证与测试起点为 2025-02-12、2025-02-19。适用范围是短期震群。[作者档案](https://zenodo.org/records/23059305)

四套目录保留原始震级标度，门槛为固定研究筛选。各自通过作者文件校验、源与缓存物理字段对照、float64 计分间隔检查和真实 DataLoader 加载。智利两套的建模时间窗分开。西希腊和 Santorini 分别覆盖不同的观测窗和区域，希腊 NOA 的原时间覆盖问题另行记录。

原始文件与处理表位于 `/root/autodl-fs/em_eqf_catalogs/doi_expansion_20261106/`。四套逐目录报告位于 `experiments/revision/catalog_doi_expansion_20261106/reports/`。标准化入口为 `scripts/data/prepare_doi_expansion_20261106.py`，可传入表中去掉 `-Standard` 的目录名。

另已取得三份原档，暂作专项数据。死海断裂带 25,000 条为多机构统一震级汇编，包含转换震级及潜在矿区事件，需进一步核对分段覆盖和类型。玻利维亚 Cochabamba 的 ISF 有 438 个事件，来自 2022 年六个月的临时台网。Gakkel 海岭有 1,225 条，原表时区待核，深度集中在 10/20 km，且原表首年 2012 与网页标题 2013 存在差异。三份文件同在该批 `raw/` 下。[死海断裂带](https://doi.org/10.31905/UQUUGRGH) · [玻利维亚](https://doi.org/10.31905/CBEE1FOR) · [Gakkel](https://doi.org/10.31905/SMUPNWEP)

## 此前区域目录

| 目录 | 标准记录 | 筛选下限 | 下载来源与处理 |
|---|---:|---:|---|
| JMA Kyushu / West Honshu / East Honshu / Hokkaido | 6,851 / 3,252 / 80,931 / 10,245 | 3.0 | JMA 1998–2024 年度 `hYYYY.zip`，原件在 `data/JMA/raw/`。按 96 字节记录解码并将 JST 转 UTC。取气象厅自然地震、人工质量 K/k、有效定位方法，再以左闭右开的地理框分区。[年度文件](https://www.data.jma.go.jp/eqev/data/bulletin/hypo_e.html) · [格式和质量码](https://www.data.jma.go.jp/eqev/data/bulletin/data/format/hypfmt_e.html) |
| GeoNet | 45,426 | 3.0 | 取 GeoNet CSV 快照，明确 earthquake、manual、confirmed/reviewed/final，至少 5 台站、gap≤180°、RMS≤1 s。FDSN 与 CSV 的逐年差异见审计。[CSV 查询](https://quakesearch.geonet.org.nz/) · [字段说明](https://www.geonet.org.nz/data/types/eq_catalogue) |
| INGV | 44,161 | 2.5 | ISIDe 分窗下载并按事件 ID 对应首选 QuakeML 震源，取明确 earthquake 且首选震源 manual、reviewed/final。[ISIDe](https://data.ingv.it/en/dataset/09) · [事件接口](https://webservices.ingv.it/fdsnws/event/1/) |
| NCEDC | 28,813 | 2.5 | 公共云年度 CSV，按明确 eq/earthquake、至少 8 台站、gap≤180°、RMS≤0.5 s、水平误差≤2 km、深度误差≤5 km。原有状态码随记录保留。[年度归档](https://ncedc.org/db/cloud.html) |
| USGS Hawaii / Puerto Rico / Alaska | 14,366 / 2,659 / 15,841 | 2.5 / 2.5 / 3.0 | FDSN 按 `hv` / `pr` / `ak` 贡献者、区域、earthquake、reviewed 下载。夏威夷和波多黎各使用完整数值筛选，阿拉斯加主筛选为 RMS≤1 s、深度误差≤5 km，完整水平定位验证另列 `spatial_quality_verified`。[贡献者目录](https://earthquake.usgs.gov/data/comcat/catalog/) |
| SED Switzerland | 993 | 2.0 | 瑞士 SED QuakeML，首选震源及震级，明确自然地震，manual 和 confirmed/reviewed/final，加完整数值定位筛选。[SED 事件服务](https://www.seismo.ethz.ch/en/knowledge/earthquake-data-and-analysis-tools/fdsn-web-services/) |
| USGS PNW / Utah / New Madrid / Montana | 1,330 / 3,196 / 1,243 / 2,914 | 2.0 | USGS FDSN 的 `uw` / `uu` / `nm` / `mb`，按区域、earthquake、reviewed 下载并加完整数值定位筛选。保存实际首选机构及震级类型。[查询接口](https://earthquake.usgs.gov/fdsnws/event/1/) |
| NOA Greece | 13,145 | 2.5 | 2014–2025 年 144 个来源月均通过覆盖核验。2018–2025 年标准建模窗口使用首选震源、人工且 reviewed/confirmed/final 的地震、完整数值定位筛选。训练 / 验证 / 测试为 124 / 4,927 / 8,094。[NOA FDSN](https://eida.gein.noa.gr/fdsnws/event/1/) |
| NORWAY SW | 619 | 0.0 | 作者 HypoDD 相对重定位。按同包 Nordic 记录对照事件编号、时间和 ML。至少 8 个差分约束且对应差分 RMS≤0.5 s。差分质量指标按作者方法解释。[ISC 作者条目](https://doi.org/10.31905/C63YMAEK) |
| NORWAY Bitdalsvatn | 648 | 0.0 | 作者 QuakeML 的首选震源和震级，明确 earthquake、ML≥0、水平不确定度 (0,2] km、深度不确定度 (0,5] km。保留首选解的原始缺测字段。[ISC 作者条目](https://doi.org/10.31905/6V0YYDFA) |

完整数值筛选指至少 8 台站、gap≤180°、RMS≤1 s、报告水平误差≤2 km、深度误差≤5 km，NCEDC 的 RMS 使用表中更严格门槛。所有震级门槛均为当前固定研究选择。时变完备性需在对应训练期、地区和震级类型下估计。

首批 JMA、GeoNet、INGV、NCEDC 的默认验证与测试起点为 2014-01-01、2016-01-01，JMA 起止按档案 JST 年界换算。随后 USGS 和瑞士默认观测 2014–2025 年，验证与测试起点为 2019-01-01、2022-01-01。两个挪威目录使用各自观测期和切分。精确边界与实际计数见各批 `reports/summary.json` 和配置。跨区独立性需考虑瑞士/INGV、PNW/NCEDC、Utah/Montana 以及两个挪威目录中的候选同震。

## EarthquakeNPP 与项目历史输入

EarthquakeNPP 作者仓库固定到提交 `26d18048e1ca8ff2b02c7016b993de48ed0760f5`。下载原 CSV、作者说明和区域形状文件，记录 Git blob 与 SHA-256，按作者 UTC 日期协议和投影坐标生成各自缓存。源表缺少深度，序列保存时间、震级和二维位置。记录数包含历史上下文。[作者数据目录与切分协议](https://github.com/ss15859/EarthquakeNPP/blob/26d18048e1ca8ff2b02c7016b993de48ed0760f5/Datasets/README.md)

| 基准 | 筛选下限 | 记录数 |
|---|---:|---:|
| `ENPP_ComCat25` | 2.5 | 92,263 |
| `ENPP_SCEDC20` / `ENPP_SCEDC25` / `ENPP_SCEDC30` | 2.0 / 2.5 / 3.0 | 130,352 / 43,584 / 12,988 |
| `ENPP_QTMSanJacinto10` / `ENPP_QTMSaltonSea10` | 1.0 | 21,291 / 45,570 |
| `ENPP_White06` | 0.6 | 58,636 |
| `ENPP_JapanDeprecated`，旧实验兼容 | 2.5 | 37,581 |

三档 SCEDC 是嵌套样本。旧 QTM 加载器的来源是 SCEDC 12 MAD 原始目录，与上表固定的作者基准快照分别管理。旧 White 适配器的文件 URL 可在 `src/catalogs/white.py` 查到。[SCEDC QTM](https://scedc.caltech.edu/eq-catalogs/qtm.html)

- **川滇历史实验。** 复用项目继承的 `data/ChuanDian/raw/chuandian_2021.csv`，同目录含 `chuandian_2021.dat`。当前可追溯的入口是这份冻结文件，原始发布下载地址待补。川滇与 SCEDC 对照按 M≥3，观测期 `[2000-01-01, 2021-05-24)`，验证与测试起点分别为 2015-01-01、2018-01-01。
- **SCEDC 历史目标实验。** 从 [SCEDC 年度目录](https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/) 取得的 SCSN 数据另存为 `data/SCEDC/raw/scsn_eq_only_snapshot_20261102.csv`，实际目标构建校验输入哈希与 `ET=eq`，取 M≥2.5，观测期 `[1985-01-01, 2020-01-01)`，验证与测试起点分别为 2005-01-01、2014-01-01。两个历史目标的构建入口是 `scripts/data/prepare_chuandian_scedc_tpp.py`，冻结结果的清单是 `experiments/revision/multiregion_pretraining/reports/chuandian_scedc_tpp_20261105/data.json`。
- **Geysers 等继承的诱发地震实验。** 项目原始输入已在 `data/Geysers/raw/`，含地震目录、注入和产出表。实际处理入口为 `scripts/data/build_geysers_processed.py`，各注入场址与自然地震目录分开管理。继承文件的外部发布版本以原项目记录为准。

## 暂存与待核对

| 数据 | 已核查的状态 |
|---|---|
| IPOC-Deep | 2007–2023 原 CSV 已下载，约 227 万条。A/B 且 M≥2.3 的筛选快照 42,660 条，源时间字段的 UTC 偏移待确认。原件在 `data/IPOCDeep/raw/`。[GFZ DOI](https://doi.org/10.5880/fidgeo.2026.050) |
| 冰岛 ISOR Hengill | 人工参考目录已下载，严格质量筛选后 19 条，验证与测试窗样本量不足。[COSEISMIQ 目录](https://www.coseismiq.ethz.ch/en/dissemination/catalogs/) |
| Sørfjorden / Maule | Sørfjorden 1,833 条，源时区及质量定义待核。Maule 仅有定位字段，当前留作位置分析，震级待补。[Sørfjorden](https://doi.org/10.31905/JA89PW86) · [Maule](https://doi.org/10.31905/ODY1A8PS) |

## 处理流程与复现

1. 保存原始响应及来源 URL、版本或查询参数，记录哈希。ZIP 检查 CRC，作者仓库按提交与 blob 固定，接口分窗保留覆盖清单。遇到空窗口时区分真实零事件和服务失败。
2. 按源文件格式还原 UTC、经纬度、深度单位和原震级类型。保留物理时间、源事件 ID 和质量字段。CWA 与 Wu 源表缺少可复用的稳定 ID，用版本内文件年份和行号生成身份。
3. 按各来源的质量协议筛选，再按区域及观测窗口生成标准源表。评估同震候选和可能的数据泄漏。相同时间但物理字段有差异的事件可保留。
4. 以左闭右开的物理时间区间拆分训练、验证、测试。归一化及参考统计使用训练期，后续区间可携带历史上下文。缓存使用 float64 时间间隔，单位为天。同刻事件用受时间分辨率约束的模型时间展开，原始物理时间留在 Parquet。
5. 逐事件核对源表和缓存，检查计分 ID、时间、震级、坐标、深度与源文件哈希。用真实 `prepare_data_tpp` 和 DataLoader 验收。精确配置保存在各缓存的 `catalog_cfg.json` 与各批次 `configs/`。

| 批次 | 处理入口，位于 `scripts/data/` | 远端报告目录，位于 `experiments/revision/` |
|---|---|---|
| JMA / GeoNet / INGV / NCEDC | `expand_catalogs_20261105.py`、`prepare_external_standard.py` | `catalog_expansion_20261105/reports/`、`catalog_standardization_20261105/reports/` |
| USGS 与 SED 第一批 | `download_regional_catalogs.py`、`prepare_regional_standard.py` | `catalog_additions_20261105/reports/` |
| USGS 与 SED 后续批次、NOA 与冰岛 | `download_more_catalogs.py`、`download_new_madrid.py`、`prepare_more_standard.py`、`resume_noa_catalog.py` | `catalog_more_20261105/reports/`、`catalog_optimization_20261105/` |
| Montana、挪威与暂存研究档案 | `normalize_research_20261106.py`、`prepare_research_standard.py` | `catalog_research_20261106/reports/` |
| EarthquakeNPP 固定版 | `download_enpp_20261106.py`、`prepare_enpp_benchmarks.py` | `catalog_benchmarks_20261106/reports/` |
| 台湾与东安纳托利亚 | `normalize_source_expansion_20261106.py`、`prepare_source_expansion_20261106.py` | `catalog_source_expansion_20261106/reports/` |

在远端项目内加载已验证缓存

```bash
cd /root/autodl-tmp/em_eqf
source /root/autodl-tmp/venvs/em_eqf-cu128/env.sh
python -c 'from src.data import Catalog; c = Catalog.by_name("Taiwan_CWA-Standard")(); print(len(c.full_sequence), c.root_dir)'
```

从本轮保存的原件复现解析、标准化和验收

```bash
cd /root/autodl-tmp/em_eqf
source /root/autodl-tmp/venvs/em_eqf-cu128/env.sh
python scripts/data/normalize_source_expansion_20261106.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/data/prepare_source_expansion_20261106.py
```

本地工作区 `D:/Doc/ntpp_transfer` 中按批次保存报告。此前的便携包分别位于 `catalog_benchmarks_20261106/earthquakenpp_benchmarks_supplement_20261106.tar.gz` 和 `catalog_research_20261106/research_catalogs_supplement_20261106.tar.gz`。每个包附原始来源、缓存、配置和 SHA-256 清单。
