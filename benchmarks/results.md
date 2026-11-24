
### 2025 test set, all: 197 games, 4782 KiB unpacked

| codec | total KiB | vs submitted zip | vs best legal zip |
|---|---:|---:|---:|
| submitted .zip (as entered) | 2123.7 | 0.0% | 8.3% |
| zip: Deflate (7-Zip, 15 passes) | 2095.2 | -1.3% | 6.8% |
| zip: Deflate via Zopfli (advzip -4) | 2053.8 | -3.3% | 4.7% |
| zip: Deflate64 | 2094.1 | -1.4% | 6.8% |
| zip: BZip2 | 2144.0 | 1.0% | 9.3% |
| zip: LZMA | 2079.4 | -2.1% | 6.0% |
| zip: PPMd (best order) | 2008.6 | -5.4% | 2.4% |
| **best legal zip per game** | 1961.0 | -7.7% | 0.0% |
| zopfli raw deflate stream (no zip headers) | 1960.3 | -7.7% | -0.0% |
| xz -9e (raw lzma2) | 1946.5 | -8.3% | -0.7% |
| brotli -q11 -w24 | 1857.3 | -12.5% | -5.3% |
| zstd -19 | 2002.5 | -5.7% | 2.1% |
| brotli + trained dict (best of 1MB zstd-trained, 16MB raw) | 1694.5 | -20.2% | -13.6% |
| zstd + trained dict (best of 1MB trained, 16MB raw --long) | 1722.6 | -18.9% | -12.2% |
| model, untrained | 1812.4 | -14.7% | -7.6% |
| model, pretrained 1 yr (2020-2020) | 1652.1 | -22.2% | -15.7% |
| model, pretrained 2 yrs (2020-2021) | 1623.5 | -23.6% | -17.2% |
| model, pretrained 3 yrs (2020-2022) | 1606.7 | -24.3% | -18.1% |
| model, pretrained 4 yrs (2020-2023) | 1606.4 | -24.4% | -18.1% |
| model, pretrained 5 yrs (2020-2024) | 1596.3 | -24.8% | -18.6% |

### 2025 test set, mostly code (<20% binary assets): 184 games, 4560 KiB unpacked

| codec | total KiB | vs submitted zip | vs best legal zip |
|---|---:|---:|---:|
| submitted .zip (as entered) | 1980.1 | 0.0% | 8.6% |
| zip: Deflate (7-Zip, 15 passes) | 1953.5 | -1.3% | 7.1% |
| zip: Deflate via Zopfli (advzip -4) | 1916.5 | -3.2% | 5.1% |
| zip: Deflate64 | 1952.4 | -1.4% | 7.0% |
| zip: BZip2 | 1998.0 | 0.9% | 9.5% |
| zip: LZMA | 1936.6 | -2.2% | 6.2% |
| zip: PPMd (best order) | 1868.5 | -5.6% | 2.4% |
| **best legal zip per game** | 1824.1 | -7.9% | 0.0% |
| zopfli raw deflate stream (no zip headers) | 1837.0 | -7.2% | 0.7% |
| xz -9e (raw lzma2) | 1822.6 | -8.0% | -0.1% |
| brotli -q11 -w24 | 1738.1 | -12.2% | -4.7% |
| zstd -19 | 1872.6 | -5.4% | 2.7% |
| brotli + trained dict (best of 1MB zstd-trained, 16MB raw) | 1582.4 | -20.1% | -13.3% |
| zstd + trained dict (best of 1MB trained, 16MB raw --long) | 1606.6 | -18.9% | -11.9% |
| model, untrained | 1692.8 | -14.5% | -7.2% |
| model, pretrained 1 yr (2020-2020) | 1538.3 | -22.3% | -15.7% |
| model, pretrained 2 yrs (2020-2021) | 1511.2 | -23.7% | -17.2% |
| model, pretrained 3 yrs (2020-2022) | 1495.1 | -24.5% | -18.0% |
| model, pretrained 4 yrs (2020-2023) | 1494.8 | -24.5% | -18.1% |
| model, pretrained 5 yrs (2020-2024) | 1485.7 | -25.0% | -18.5% |

### 2025 test set, asset-heavy (>=20% binary): 13 games, 222 KiB unpacked

| codec | total KiB | vs submitted zip | vs best legal zip |
|---|---:|---:|---:|
| submitted .zip (as entered) | 143.7 | 0.0% | 5.0% |
| zip: Deflate (7-Zip, 15 passes) | 141.7 | -1.4% | 3.5% |
| zip: Deflate via Zopfli (advzip -4) | 137.3 | -4.4% | 0.3% |
| zip: Deflate64 | 141.7 | -1.4% | 3.5% |
| zip: BZip2 | 146.0 | 1.7% | 6.7% |
| zip: LZMA | 142.8 | -0.6% | 4.4% |
| zip: PPMd (best order) | 140.1 | -2.5% | 2.4% |
| **best legal zip per game** | 136.9 | -4.7% | 0.0% |
| zopfli raw deflate stream (no zip headers) | 123.3 | -14.2% | -9.9% |
| xz -9e (raw lzma2) | 123.9 | -13.7% | -9.5% |
| brotli -q11 -w24 | 119.2 | -17.0% | -12.9% |
| zstd -19 | 129.9 | -9.6% | -5.1% |
| brotli + trained dict (best of 1MB zstd-trained, 16MB raw) | 112.2 | -21.9% | -18.0% |
| zstd + trained dict (best of 1MB trained, 16MB raw --long) | 116.0 | -19.3% | -15.3% |
| model, untrained | 119.6 | -16.7% | -12.6% |
| model, pretrained 1 yr (2020-2020) | 113.8 | -20.8% | -16.8% |
| model, pretrained 2 yrs (2020-2021) | 112.3 | -21.8% | -17.9% |
| model, pretrained 3 yrs (2020-2022) | 111.6 | -22.3% | -18.4% |
| model, pretrained 4 yrs (2020-2023) | 111.6 | -22.3% | -18.4% |
| model, pretrained 5 yrs (2020-2024) | 110.5 | -23.1% | -19.2% |

Median per-game ratio vs best legal zip: model untrained 0.925, pretrained 5 yrs 0.789, brotli+dict 0.885, zstd+dict 0.906
Capacity: a 13,312-byte archive.13k holds about 16.5 KiB of best-zip-equivalent game with the pretrained model.
