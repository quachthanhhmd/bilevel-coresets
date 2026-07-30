import numpy as np
import torch
import torch.nn.functional as F


class Training():

    def __init__(self, model, device, nr_epochs, beta=1):
        self.model = model
        self.device = device
        self.nr_epochs = nr_epochs
        self.beta = beta
        self.buffer = []

    def train(self, train_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5 * 1e-4)
        self.model.train()
        for epoch in range(1, self.nr_epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                optimizer.step()

    def loss(self, output, target):
        loss = F.cross_entropy(output, target)
        if self.buffer:
            # Gộp TẤT CẢ các slot của buffer thành một khối duy nhất rồi tính MỘT
            # trung bình có trọng số CHUẨN HOÁ (chia cho tổng trọng số), thay vì cộng
            # dồn beta * mean(...) riêng cho từng slot.
            #
            # Vì sao: reservoir/cbrs lưu buffer dưới dạng 1 slot duy nhất, trọng số
            # luôn = 1 -> vòng lặp cũ chỉ chạy 1 lần, mean(loss * 1) = mean(loss), không
            # vấn đề gì. Nhưng buffer merge-reduce của coreset gồm nhiều slot (mặc định
            # nr_slots=10), và mỗi lần merge_reduce() gộp 2 slot, trọng số slot mới =
            # tổng trọng số 2 slot cũ (tăng dần không giới hạn qua thời gian, không được
            # chuẩn hoá lại). Với code cũ, replay-loss của coreset bị cộng dồn
            # beta * mean(loss * w) NHIỀU LẦN (một lần mỗi slot) với w có thể đã tăng
            # rất lớn qua nhiều lượt gộp -- tổng hệ số khuếch đại thực tế trở thành
            # beta * nr_slots * (trọng số tích luỹ), lớn hơn rất nhiều so với beta đơn
            # thuần của reservoir/cbrs, khiến replay-loss áp đảo hoàn toàn loss của batch
            # hiện tại (đặc biệt rõ khi beta lớn). Sửa lại bằng cách gộp toàn bộ buffer
            # và chuẩn hoá đúng nghĩa "trung bình có trọng số" giúp hệ số khuếch đại chỉ
            # còn phụ thuộc vào beta, nhất quán với cách reservoir/cbrs hoạt động.
            all_X = np.concatenate([data[0] for data, _ in self.buffer])
            all_y = np.concatenate([data[1] for data, _ in self.buffer])
            all_w = np.concatenate([w for _, w in self.buffer])

            cs_data = torch.from_numpy(all_X).to(self.device).type(torch.float)
            cs_target = torch.from_numpy(all_y).to(self.device).type(torch.long)
            cs_w = torch.from_numpy(all_w).type(torch.float).to(self.device)
            cs_output = self.model(cs_data)
            per_point_loss = F.cross_entropy(cs_output, cs_target, reduction='none')
            loss += self.beta * torch.sum(per_point_loss * cs_w) / torch.sum(cs_w)
        return loss

    def test(self, test_loader):
        self.model.eval()
        correct = 0
        loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.cross_entropy(output, target, reduction='sum').cpu().item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(test_loader.dataset)
