import numpy as np
import torch, torchvision
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler


def MNIST(data_path, batch_size, shuffle=False, train=True, condition_on=None, num_workers=0, rescale_to=64, holdout=False):
	img_size, num_channels = 28, 1
	img_size_scaled = rescale_to
	transform = torchvision.transforms.Compose([
		#torchvision.transforms.Scale(img_size_scaled),
		#torchvision.transforms.CenterCrop(img_size_scaled),
		torchvision.transforms.ToTensor()
		])
	dataset = torchvision.datasets.MNIST(data_path, train, download=False, transform=transform)

	if condition_on is not None:
		# sample full dataset once, determine which samples belong to conditioned class
		sampler = SequentialSampler(dataset)
		data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=len(dataset), shuffle=False, num_workers=num_workers)
		data_iter = iter(data_loader)
		_, labels = data_iter.next()
		remove_label = np.in1d(labels.numpy().ravel(), condition_on)
		ids = np.where(np.in1d(labels.numpy().ravel(), condition_on))[0]

		if not holdout:
			# sample randomly without replacement from conditioned class
			#sampler = SubsetRandomSampler(ids)
			sampler = SequentialSampler(ids)
			return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

		else:
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			#sampler_train = SubsetRandomSampler(ids_train)
			#sampler_holdout = SubsetRandomSampler(ids_holdout)
			sampler_train = SequentialSampler(ids_train)
			sampler_holdout = SequentialSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, shuffle=False, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, shuffle=False, num_workers=num_workers), img_size_scaled, num_channels

	else:
		if not holdout:
			return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

		else:
			ids = np.arange(0, len(dataset))
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			#sampler_train = SubsetRandomSampler(ids_train)
			#sampler_holdout = SubsetRandomSampler(ids_holdout)
			sampler_train = SequentialSampler(ids_train)
			sampler_holdout = SequentialSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

def FMNIST(data_path, batch_size, shuffle=False, train=True, condition_on=None, num_workers=0, rescale_to=64, holdout=False):
	img_size, num_channels = 28, 1
	img_size_scaled = rescale_to
	transform = torchvision.transforms.Compose([
		#torchvision.transforms.Scale(img_size_scaled),
		#torchvision.transforms.CenterCrop(img_size_scaled),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	dataset = torchvision.datasets.FashionMNIST(data_path, train, download=False, transform=transform)
	print("ALLdataset=>",len(dataset))

	if condition_on is not None:
		print("digit == True")
		# sample full dataset once, determine which samples belong to conditioned class
		sampler = SequentialSampler(dataset)
		data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=len(dataset), shuffle=False, num_workers=num_workers)
		data_iter = iter(data_loader)
		_, labels = data_iter.next()
		#print("label",labels)
		anomaly_label = np.in1d(labels.numpy().ravel(), condition_on)
		train_label = np.logical_not(np.in1d(labels.numpy().ravel(), condition_on))
		#print(f"anomaly_label=>",anomaly_label)
		#print(f"train_label=>",train_label)
		
		train_ids = np.where(np.logical_not(np.in1d(labels.numpy().ravel(), condition_on)))[0]
		anomaly_ids = np.where(np.in1d(labels.numpy().ravel(), condition_on))[0]

		if not holdout:
			# sample randomly without replacement from conditioned class
			tr_sampler = SubsetRandomSampler(train_ids)
			an_sampler = SubsetRandomSampler(anomaly_ids)
			train_dataset = torch.utils.data.DataLoader(dataset, sampler=tr_sampler, batch_size=128, shuffle=False, num_workers=num_workers)
			anomaly_dataset = torch.utils.data.DataLoader(dataset, sampler=an_sampler, batch_size=128, shuffle=False, num_workers=num_workers)
			print("====================")
			print("TrainData=>",len(train_dataset))
			print("AnomalyData=>",len(anomaly_dataset))
			print("====================")
			
			return train_dataset, anomaly_dataset, img_size_scaled, num_channels, 

		else:
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			#sampler_train = SubsetRandomSampler(ids_train)
			#sampler_holdout = SubsetRandomSampler(ids_holdout)
			sampler_train = SequentialSampler(ids_train)
			sampler_holdout = SequentialSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

	else:
		if not holdout:
			return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

		else:
			prine("test")
			
			ids = np.arange(0, len(dataset))
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			sampler_train = SequentialSampler(ids_train)
			sampler_holdout = SequentialSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, shuffle=False, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels


def CIFAR10(data_path, batch_size, shuffle=True, train=True, condition_on=None, num_workers=0, rescale_to=29, holdout=False):
	img_size, num_channels = 32, 3
	img_size_scaled = rescale_to
	transform = torchvision.transforms.Compose([
		#torchvision.transforms.Scale(img_size_scaled),
		#torchvision.transforms.CenterCrop(img_size_scaled),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	dataset = torchvision.datasets.CIFAR10(data_path, train, download=False, transform=transform)
	print("ALLdataset=>",len(dataset))

	if condition_on is not None:
		print("digit == True")
		# sample full dataset once, determine which samples belong to conditioned class
		sampler = SequentialSampler(dataset)
		data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=len(dataset), shuffle=False, num_workers=num_workers)
		data_iter = iter(data_loader)
		_, labels = data_iter.next()
		#print("label",labels)
		anomaly_label = np.in1d(labels.numpy().ravel(), condition_on)
		train_label = np.logical_not(np.in1d(labels.numpy().ravel(), condition_on))
		#print(f"anomaly_label=>",anomaly_label)
		#print(f"train_label=>",train_label)
		
		train_ids = np.where(np.logical_not(np.in1d(labels.numpy().ravel(), condition_on)))[0]
		anomaly_ids = np.where(np.in1d(labels.numpy().ravel(), condition_on))[0]

		if not holdout:
			# sample randomly without replacement from conditioned class
			tr_sampler = SubsetRandomSampler(train_ids)
			an_sampler = SubsetRandomSampler(anomaly_ids)
			train_dataset = torch.utils.data.DataLoader(dataset, sampler=tr_sampler, batch_size=1024, shuffle=False, num_workers=num_workers)
			anomaly_dataset = torch.utils.data.DataLoader(dataset, sampler=an_sampler, batch_size=1024, shuffle=False, num_workers=num_workers)
			print("====================")
			print("TrainData=>",len(train_dataset))
			print("AnomalyData=>",len(anomaly_dataset))
			print("====================")
			
			return train_dataset, anomaly_dataset, img_size_scaled, num_channels, 

		else:
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			#sampler_train = SubsetRandomSampler(ids_train)
			#sampler_holdout = SubsetRandomSampler(ids_holdout)
			sampler_train = SequentialSampler(ids_train)
			sampler_holdout = SequentialSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

	else:
		if not holdout:
			return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

		else:
			prine("test")
			
			ids = np.arange(0, len(dataset))
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			sampler_train = SequentialSampler(ids_train)
			sampler_holdout = SequentialSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, shuffle=False, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

def STL10(data_path, batch_size, shuffle=True, train=True, condition_on=None, num_workers=0, rescale_to=64, holdout=False):
	img_size, num_channels = 96, 3
	img_size_scaled = rescale_to
	transform = torchvision.transforms.Compose([
		#torchvision.transforms.Scale(img_size_scaled),
		#torchvision.transforms.CenterCrop(img_size_scaled),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	dataset = torchvision.datasets.STL10(data_path, train, download=True, transform=transform)
	print("ALLdataset=>",len(dataset))

	if condition_on is not None:
		print("digit == True")
		# sample full dataset once, determine which samples belong to conditioned class
		sampler = SequentialSampler(dataset)
		data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=len(dataset), shuffle=False, num_workers=num_workers)
		data_iter = iter(data_loader)
		_, labels = data_iter.next()
		#print("label",labels)
		anomaly_label = np.in1d(labels.numpy().ravel(), condition_on)
		train_label = np.logical_not(np.in1d(labels.numpy().ravel(), condition_on))
		#print(f"anomaly_label=>",anomaly_label)
		#print(f"train_label=>",train_label)
		
		train_ids = np.where(np.logical_not(np.in1d(labels.numpy().ravel(), condition_on)))[0]
		anomaly_ids = np.where(np.in1d(labels.numpy().ravel(), condition_on))[0]

		if not holdout:
			# sample randomly without replacement from conditioned class
			tr_sampler = SubsetRandomSampler(train_ids)
			an_sampler = SubsetRandomSampler(anomaly_ids)
			train_dataset = torch.utils.data.DataLoader(dataset, sampler=tr_sampler, batch_size=128, shuffle=False, num_workers=num_workers)
			anomaly_dataset = torch.utils.data.DataLoader(dataset, sampler=an_sampler, batch_size=128, shuffle=False, num_workers=num_workers)
			print("====================")
			print("TrainData=>",len(train_dataset))
			print("AnomalyData=>",len(anomaly_dataset))
			print("====================")
			
			return train_dataset, anomaly_dataset, img_size_scaled, num_channels, 

		else:
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			#sampler_train = SubsetRandomSampler(ids_train)
			#sampler_holdout = SubsetRandomSampler(ids_holdout)
			sampler_train = SequentialSampler(ids_train)
			sampler_holdout = SequentialSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

	else:
		if not holdout:
			return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

		else:
			prine("test")
			
			ids = np.arange(0, len(dataset))
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			sampler_train = SequentialSampler(ids_train)
			sampler_holdout = SequentialSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, shuffle=False, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

