# Learned Compression of Encoding Distributions

<table style="text-align: center">
 <tr>
  <td width="400px"><img src="https://github.com/multimedialabsfu/learned-compression-of-encoding-distributions/assets/721196/d6d9b6da-95ca-4b00-b34c-c907a58528af" /><br /><sup>This shows the suboptimality of using a single static encoding distribution. This distribution is optimal, on average, among all static distributions, but it is suboptimal for any specific sample.</sup></td>
  <td width="400px"><img src="https://github.com/multimedialabsfu/learned-compression-of-encoding-distributions/assets/721196/f323a7fc-ac77-4b68-b898-e13bb4387380" /><br /><sup>Proposed adaptive encoding distribution architecture.</sup></td>
 </tr>
 <tr>
  <td width="400px"><img src="https://github.com/multimedialabsfu/learned-compression-of-encoding-distributions/assets/721196/8cc03704-f39a-4786-90dc-ccee9f2927e7" /><br /><sup>Visualization of target (p) and reconstructed (p̂) encoding distributions. Our proposed method reconstructs p̂, which is then used by the fully-factorized entropy model to encode the latent derived from a given input image. Each collection of distributions is visualized as a color plot, with channels varying along the x-axis, bins varying along the y-axis, and negative log-likelihoods represented by the z-axis (i.e., color). </sup></td>
  <td width="400px"><img src="https://github.com/multimedialabsfu/learned-compression-of-encoding-distributions/assets/721196/1dc79c87-1717-467f-a013-cd9c2afe4af2" /><br /><sup>RD curves for the Kodak dataset. The same pretrained $g_a$ and $g_s$ transform weights were used across the various methods for better direct comparison. For the fully-factorized architecture — where each channel is paired with a encoding distribution — our adaptive method comes close to the ideal performance achieved by using the best encoding distribution for a given image.</sup></td>
 </tr>
</table>

> <sup>**Abstract:** The entropy bottleneck introduced by Ballé et al. is a common component used in many learned compression models. It encodes a transformed latent representation using a static distribution whose parameters are learned during training. However, the actual distribution of the latent data may vary wildly across different inputs. The static distribution attempts to encompass all possible input distributions, thus fitting none of them particularly well. This unfortunate phenomenon, sometimes known as the amortization gap, results in suboptimal compression. To address this issue, we propose a method that dynamically adapts the encoding distribution to match the latent data distribution for a specific input. First, our model estimates a better encoding distribution for a given input. This distribution is then compressed and transmitted as an additional side-information bitstream. Finally, the decoder reconstructs the encoding distribution and uses it to decompress the corresponding latent data. Our method achieves a Bjøntegaard-Delta (BD)-rate gain of -7.10% on the Kodak test dataset when applied to the standard fully-factorized architecture. Furthermore, considering computational complexity, the transform used by our method is an order of magnitude cheaper in terms of Multiply-Accumulate (MAC) operations compared to related side-information methods such as the scale hyperprior.</sup>

- **Authors:** Mateen Ulhaq and Ivan V. Bajić
- **Affiliation:** Simon Fraser University
- **Links:** Accepted at ICIP 2024. [[Paper][arXiv]]. [[BibTeX citation](#citation)].


----


## Citation

Please cite this work as:

```bibtex
@inproceedings{ulhaq2024encodingdistributions,
  title = {Learned Compression of Encoding Distributions},
  author = {Ulhaq, Mateen and Baji\'{c}, Ivan V.},
  booktitle = {Proc. IEEE ICIP},
  year = {2024},
}
```




[arXiv]: https://arxiv.org/abs/2406.13059
[Slides]: https://raw.githubusercontent.com/multimedialabsfu/learned-compression-of-encoding-distributions/assets/main/assets/slides.pdf
[CompressAI Trainer]: https://github.com/InterDigitalInc/CompressAI-Trainer
[walkthrough]: https://interdigitalinc.github.io/CompressAI-Trainer/tutorials/full.html
