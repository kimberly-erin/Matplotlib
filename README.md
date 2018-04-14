
Observation 1:  Capomulin was the only drug able to reduce the size of the tumor.

Observation 2: Capomulin had the best success at reducing metastic sites, important for cancers with circulating tumor cells.

Observation 3:  These two combined attributes contribut to Capumolin having the best survival rate.



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
```


```python
# Import data into data frames
pharma_data = pd.read_csv("raw_data/clinicaltrial_data.csv")
mouse_data = pd.read_csv("raw_data/mouse_drug_data.csv")

print(pharma_data.head())
print(mouse_data.head())
```

      Mouse ID  Timepoint  Tumor Volume (mm3)  Metastatic Sites
    0     b128          0                45.0                 0
    1     f932          0                45.0                 0
    2     g107          0                45.0                 0
    3     a457          0                45.0                 0
    4     c819          0                45.0                 0
      Mouse ID      Drug
    0     f234  Stelasyn
    1     x402  Stelasyn
    2     a492  Stelasyn
    3     w540  Stelasyn
    4     v764  Stelasyn
    


```python
# Combine data frames
combined_data = pd.merge(pharma_data,mouse_data,on="Mouse ID")

drugs = combined_data["Drug"].unique()
days = combined_data["Timepoint"].unique()

selected=["Capomulin", "Infubinol", "Ketapril", "Placebo"]
filtered_data=combined_data[combined_data['Drug'].isin(selected)]

filtered_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
      <th>Drug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b128</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b128</td>
      <td>5</td>
      <td>45.651331</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b128</td>
      <td>10</td>
      <td>43.270852</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b128</td>
      <td>15</td>
      <td>43.784893</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b128</td>
      <td>20</td>
      <td>42.731552</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Volume changes over time by drug
drug_df = filtered_data.groupby(["Drug","Timepoint"])

tumor_vol = drug_df["Tumor Volume (mm3)"].mean()

drug_affect_vol = pd.DataFrame({"Tumor Volume (mm3)": tumor_vol})
drug_affect_vol.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Tumor Volume (mm3)</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Capomulin</th>
      <th>0</th>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>44.266086</td>
    </tr>
    <tr>
      <th>10</th>
      <td>43.084291</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42.064317</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40.716325</td>
    </tr>
  </tbody>
</table>
</div>




```python
#for each drug in drugs, average of Tumer Volume over time

size_unstacked = drug_affect_vol.unstack(level=-2, fill_value=None)
size_unstacked
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">Tumor Volume (mm3)</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>44.266086</td>
      <td>47.062001</td>
      <td>47.389175</td>
      <td>47.125589</td>
    </tr>
    <tr>
      <th>10</th>
      <td>43.084291</td>
      <td>49.403909</td>
      <td>49.582269</td>
      <td>49.423329</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42.064317</td>
      <td>51.296397</td>
      <td>52.399974</td>
      <td>51.359742</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40.716325</td>
      <td>53.197691</td>
      <td>54.920935</td>
      <td>54.364417</td>
    </tr>
    <tr>
      <th>25</th>
      <td>39.939528</td>
      <td>55.715252</td>
      <td>57.678982</td>
      <td>57.482574</td>
    </tr>
    <tr>
      <th>30</th>
      <td>38.769339</td>
      <td>58.299397</td>
      <td>60.994507</td>
      <td>59.809063</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37.816839</td>
      <td>60.742461</td>
      <td>63.371686</td>
      <td>62.420615</td>
    </tr>
    <tr>
      <th>40</th>
      <td>36.958001</td>
      <td>63.162824</td>
      <td>66.068580</td>
      <td>65.052675</td>
    </tr>
    <tr>
      <th>45</th>
      <td>36.236114</td>
      <td>65.755562</td>
      <td>70.662958</td>
      <td>68.084082</td>
    </tr>
  </tbody>
</table>
</div>




```python
cap = size_unstacked.iloc[:,0]
cap_sem = size_unstacked.iloc[:,0].sem()
inf = size_unstacked.iloc[:,1]
inf_sem=size_unstacked.iloc[:,1].sem()
ket = size_unstacked.iloc[:,2]
ket_sem=size_unstacked.iloc[:,2].sem()
plac = size_unstacked.iloc[:,3]
plac_sem=size_unstacked.iloc[:,3].sem()

fig, (ax) = plt.subplots()
fig.suptitle("Tumor Size in mm3 over Time", fontsize=16, fontweight="bold")

plt.rcParams["figure.figsize"] = [9,5]

ax.errorbar(days, cap, cap_sem, linewidth=1, marker="x", color = "skyblue", capsize=5, elinewidth=2, markeredgewidth=2)
ax.errorbar(days, inf, inf_sem, linewidth=1, marker="o", color = "navy", capsize=5, elinewidth=2, markeredgewidth=2)
ax.errorbar(days, ket, ket_sem, linewidth=1, marker="*", color = "blue", capsize=5, elinewidth=2, markeredgewidth=2)
ax.errorbar(days, plac, plac_sem, linewidth=1, marker="^", color = "red", capsize=5, elinewidth=2, markeredgewidth=2)

ax.set_xlim(0, 45)
ax.set_xlabel("days")
ax.set_ylabel("Volume (mm3)")
ax.legend(loc="upper left")


plt.show()
```


![png](output_6_0.png)



```python
# Metastic Site changes over time by drug
drug_df = filtered_data.groupby(["Drug","Timepoint"])

met_sites = drug_df["Metastatic Sites"].mean()

drug_affect_sites = pd.DataFrame({"Metastatic Sites": met_sites})

sites_unstacked = drug_affect_sites.unstack(level=-2, fill_value=None)
sites_unstacked
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">Metastatic Sites</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.160000</td>
      <td>0.280000</td>
      <td>0.304348</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.320000</td>
      <td>0.666667</td>
      <td>0.590909</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.375000</td>
      <td>0.904762</td>
      <td>0.842105</td>
      <td>1.250000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.652174</td>
      <td>1.050000</td>
      <td>1.210526</td>
      <td>1.526316</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.818182</td>
      <td>1.277778</td>
      <td>1.631579</td>
      <td>1.941176</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.090909</td>
      <td>1.588235</td>
      <td>2.055556</td>
      <td>2.266667</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.181818</td>
      <td>1.666667</td>
      <td>2.294118</td>
      <td>2.642857</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.380952</td>
      <td>2.100000</td>
      <td>2.733333</td>
      <td>3.166667</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1.476190</td>
      <td>2.111111</td>
      <td>3.363636</td>
      <td>3.272727</td>
    </tr>
  </tbody>
</table>
</div>




```python
cap = sites_unstacked.iloc[:,0]
cap_sem = sites_unstacked.iloc[:,0].sem()
inf = sites_unstacked.iloc[:,1]
inf_sem=sites_unstacked.iloc[:,1].sem()
ket = sites_unstacked.iloc[:,2]
ket_sem=sites_unstacked.iloc[:,2].sem()
plac = sites_unstacked.iloc[:,3]
plac_sem=sites_unstacked.iloc[:,3].sem()

fig, (ax) = plt.subplots()
fig.suptitle("Metastic Sites Over Time", fontsize=16, fontweight="bold")

plt.rcParams["figure.figsize"] = [9, 6]

ax.errorbar(days, cap, cap_sem, linewidth=1, marker="x", color = "skyblue", capsize=5, elinewidth=2, markeredgewidth=2)
ax.errorbar(days, inf, inf_sem, linewidth=1, marker="o", color = "navy", capsize=5, elinewidth=2, markeredgewidth=2)
ax.errorbar(days, ket, ket_sem, linewidth=1, marker="*", color = "blue", capsize=5, elinewidth=2, markeredgewidth=2)
ax.errorbar(days, plac, plac_sem, linewidth=1, marker="^", color = "red", capsize=5, elinewidth=2, markeredgewidth=2)

ax.set_xlim(0, 45)
ax.set_xlabel("days")
ax.set_ylabel("Metastic Sites")
ax.legend(loc="upper left")


plt.show()
```


![png](output_8_0.png)



```python
# Survival Rate
drug_df = filtered_data.groupby(["Drug","Timepoint"])

survival_rate = drug_df["Mouse ID"].count()/25*100

survival_df = pd.DataFrame({"Mouse Count": survival_rate})

mouse_unstacked = survival_df.unstack(level=-2, fill_value=None)

mouse_unstacked
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">Mouse Count</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100.0</td>
      <td>100.0</td>
      <td>92.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>100.0</td>
      <td>84.0</td>
      <td>88.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>96.0</td>
      <td>84.0</td>
      <td>76.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>92.0</td>
      <td>80.0</td>
      <td>76.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>88.0</td>
      <td>72.0</td>
      <td>76.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>88.0</td>
      <td>68.0</td>
      <td>72.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>88.0</td>
      <td>48.0</td>
      <td>68.0</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>84.0</td>
      <td>40.0</td>
      <td>60.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>84.0</td>
      <td>36.0</td>
      <td>44.0</td>
      <td>44.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cap = mouse_unstacked.iloc[:,0]
cap_sem = mouse_unstacked.iloc[:,0].sem()
inf = mouse_unstacked.iloc[:,1]
inf_sem=mouse_unstacked.iloc[:,1].sem()
ket = mouse_unstacked.iloc[:,2]
ket_sem=mouse_unstacked.iloc[:,2].sem()
plac = mouse_unstacked.iloc[:,3]
plac_sem=mouse_unstacked.iloc[:,3].sem()

fig, (ax) = plt.subplots()
fig.suptitle("Survival Rates Over Time", fontsize=16, fontweight="bold")

plt.rcParams["figure.figsize"] = [9,6]

ax.errorbar(days, cap, cap_sem, linewidth=1, marker="x", color = "skyblue", capsize=5, elinewidth=2, markeredgewidth=2)
ax.errorbar(days, inf, inf_sem, linewidth=1, marker="o", color = "navy", capsize=5, elinewidth=2, markeredgewidth=2)
ax.errorbar(days, ket, ket_sem, linewidth=1, marker="*", color = "blue", capsize=5, elinewidth=2, markeredgewidth=2)
ax.errorbar(days, plac, plac_sem, linewidth=1, marker="^", color = "red", capsize=5, elinewidth=2, markeredgewidth=2)

ax.set_xlim(0, 45)
ax.set_xlabel("days")
ax.set_ylabel("Survival Rate (%)")
ax.legend(loc="lower left")


plt.show()
```


![png](output_10_0.png)



```python
#Total change in tumor
final_size = filtered_data.loc[combined_data["Timepoint"] == 45,:]

size_drug= final_size.groupby(["Drug"])

vol_change = size_drug["Tumor Volume (mm3)"].mean()-45
positive = vol_change > 0
vol_diff = pd.DataFrame({"% Change in Volume": vol_change, "Positive": positive})
vol_diff = vol_diff.replace([True,False],['red','green'])

#vol_diff["Tumor Volume (mm3)"]=vol_diff["Tumor Volume (mm3)"].map("{:.1f}".format)

vol_diff
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>% Change in Volume</th>
      <th>Positive</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capomulin</th>
      <td>-8.763886</td>
      <td>green</td>
    </tr>
    <tr>
      <th>Infubinol</th>
      <td>20.755562</td>
      <td>red</td>
    </tr>
    <tr>
      <th>Ketapril</th>
      <td>25.662958</td>
      <td>red</td>
    </tr>
    <tr>
      <th>Placebo</th>
      <td>23.084082</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>




```python
changes = vol_diff.iloc[:,0]
x_axis=np.arange(0,len(vol_diff["% Change in Volume"]),1)

fig, (ax) = plt.subplots()

plt.title("Tumor Changes Over 45 Days", fontsize=16, fontweight="bold")

plt.ylabel("% Tumor Volume Change")

plt.rcParams["figure.figsize"] = [6,4]

rects=ax.bar(x_axis, changes, color=vol_diff["Positive"].tolist())

ax.xaxis.set_ticks(np.arange(0,len(drugs),1))
ax.set_xticklabels(drugs, rotation='vertical')
plt.grid(alpha=0.25, linestyle="dashed")

plt.xlim(-0.5,3.5)


for rect in rects:
    h1 = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2., h1 / 2., "%d" % h1+"%", ha="center", va="bottom", color="white", fontsize=16, fontweight="bold")

```


![png](output_12_0.png)

