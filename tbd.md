# Experiment to do.

@author: aikenhong2022
@desc: in this file we will record those experiment we need to do next.
@purpose: improve the performance of Incremental Part >= 63.24 (now 62.99)

## Details

- [ ] Resume the config to recover 62.99 situations.

**After that** we do experiment below:

- [ ] try change the ce to the ArcFace. (only change this place)
	FIXME: still got some bug, which the loss is inf.
- [ ] change loss
  - [ ] composition from (SCL+CE_mix)-(KD) to {(CE-mix)-(kD-SCL)}
  - [x] combiner donot be started from 0. (adapt the feature extractor and classifier at the same time will be better)
  - [ ] factor from (alpha)...(1-alpha) to over "1"
- [ ] using the old std and mean (try another strategy to update the feature-mean)
- [x] using a harder transformer

**After finishing** if we cannot got better result, we may dont be compared with DER.

we should finsh most of it in the **friday**

## Analysis

we notice that the curve of training is very special,  the training phase is **end** up **early** . 
In this situation we should make it slower to reduce the overfit situation. 
One way to fix that is to change the rate between new/old more old will get a better result.

so we should:

1. Enhance the **keep** part.
2. Learning new more **earlier** to get best result before decrease.
3. Make the new task herder?

seems like not zero loss **doesnot influece** the result much. we will try using a harder transformer after this.
but i think this will not be helpful which didnot play such an important role. Which make it worse actually.

the angular loss may cause some error in calculation, we should notice the difference between us in the offical version.

