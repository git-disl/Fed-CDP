# Fed-CDP
Code for ICDCS2021 Gradient-Leakage Resilient Federated Learning.

```
Wenqi Wei, Ling Liu, Yanzhao Wu, Gong Su, and Arun Iyengar. "Gradient-Leakage Resilient Federated Learning", IEEE International Conference on Distributed Computing Systems (ICDCS 2021), Virtual, July, 2021  
```


## description

### Federated learning faces three types of gradient leakage threats basing on the place of leakage 

![threat model](description/Slide3.PNG)


### Existing approaches for federated learning with differential privacy （coined as Fed-SDP） concerns only the client-level differential privacy with per-client per-round noise.

![fed-sdp](description/Slide6.PNG)

### Our approach for federated learning with differential privacy (refered to as Fed-CDP) retrospects the instance-level differential privacy guarantee with per-example per-local iteration noise. With some differential privacy properties, the instance-level differential privacy guarantee also ensures the client-level differential privacy in federated learning.

![fed-cdp](description/Slide7.PNG)




## how to run


- <strong>FedSDP.py</strong> contains the code for training a benign and existing differentially private federated learning model (McMahan, H. Brendan, Daniel Ramage, Kunal Talwar, and Li Zhang. "Learning differentially private recurrent language models." arXiv preprint arXiv:1710.06963 (2017).). In this differentially private model, we consider Fed-SDP which adds per-client per-round noise, and only holds client level differential privacy. 

-  <strong>FedCDP.py</strong> contains the proposed Fed-CDP approach which adds per-example per-local iteration noise, instance level differential privacy. Due to the composition theorem, the instance-level noise provide both example-level and client-level differential privacy. 

- <strong>privacy_accounting_fed_clientlevel.py</strong>  and <strong>privacy_accounting_fed_instancelevel.py</strong> are codes for computing epsilon privacy spending at client level (for Fed-SDP with sampling rate = #participating clients/#total clients) and at instance level (for Fed-CDP with sampling rate = batch size * # participating client / # global data). We consider five privacy accounting methods: base composition, advanced composition, optimal composition, zCDP and Moments accountant.

- For gradient leakage attacks, please refer to our [CPL attacks](https://git-disl.github.io/ESORICS20-CPL/).




