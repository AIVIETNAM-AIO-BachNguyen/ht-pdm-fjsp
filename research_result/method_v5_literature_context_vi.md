# Literature bổ sung sau khi v5 FULL đã bắt đầu

Đọc ngày 2026-10-20, **sau** khi proposal/protocol v5 commit `83dfc8c`
và FULL đã bắt đầu. Đây là positioning note, không sửa hypothesis, source,
seeds, metric hay selection. Không dùng để diễn giải TEST theo hướng mới.

- [Zhang et al., *Learning to Dispatch for Job Shop Scheduling via Deep
  Reinforcement Learning*, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf)
  học dispatch rule từ scratch với disjunctive graph/GNN và chứng minh
  transfer sang instance lớn hơn. V5 dùng một inductive bias theo matching
  và residual resource time nhưng fixed N4/K2, nên không có claim size
  generalization tương đương.
- [Delong et al., *Online Bipartite Matching with Reusable Resources*,
  Mathematics of Operations Research 2023](https://pubsonline.informs.org/doi/abs/10.1287/moor.2022.0242)
  mô hình resource bận trong duration rồi tái sử dụng, đưa ra reranking và
  primal-dual algorithms với competitive guarantee. Vì thế duration và
  reusable-capacity trong dispatch không phải ý tưởng mới. Mô hình reward/
  arrival của họ khác persistent PM→CM, nhưng không thể suy ra v5 là method
  đầu tiên xử lý công suất còn lại.
- [Chen et al., *Online Bipartite Matching with Reusable Capacity under
  Non-Stationary Rewards*, 2026 preprint](https://arxiv.org/abs/2608.18130)
  dùng time-aware opportunity-cost loss và price từ service trajectory có
  thể bị chiếm. Đây là prior art gần hơn với lý do thiết kế v4/v5. V5 không
  claim phát minh opportunity cost hay post-decision resource valuation;
  contribution hiện tại chỉ là kiểm chứng một action-conditioned feature
  trong RL controller cho simulator request-lifecycle cụ thể. Bài báo này
  có arrival/reward và guarantee khác; chưa có đối chiếu algorithm trực
  tiếp hoặc claim v5 superior.

Nếu v5 primary dương tính, bằng chứng chỉ là internal ablation trong panel
đã khóa. Để thành nghiên cứu có novelty/transfer cần thêm đối chiếu chi
tiết với reusable matching và benchmark độc lập; không trộn kết quả này vào
phán quyết primary v5.

- [Wang & Gombolay, *Stochastic Resource Optimization over Heterogeneous
  Graph Neural Networks for Failure-Predictive Maintenance Scheduling*,
  ICAPS 2022](https://ojs.aaai.org/index.php/ICAPS/article/download/19839/19598/23852)
  là prior art **rất gần**: aircraft maintenance stochastic, shared crews,
  preventive/corrective assignment, composite action có nhiều assignment
  hoặc skip; policy factorize thành chuỗi quyết định và cập nhật graph sau
  mỗi chọn. Họ học RL trên heterogeneous graph và kiểm chứng trên AirME.
  Vì vậy không được claim RL điều phối shared maintenance crews, composite
  actions, skip hay việc đưa hậu quả assignment vào representation là mới.
  V5 khác ở việc chấm đồng thời 21 matching bằng exact current-information
  residual earliest-start feature cho queued request, không factorize chuỗi
  graph decision; đó chỉ là sự khác biệt trong thiết kế, chưa phải novelty
  hoặc superiority. AirME có stochastic service duration và aircraft
  operation; simulator v5 có deterministic technician-specific duration,
  persistent threshold PM→CM request, nên objective/score không so trực tiếp.
