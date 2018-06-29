#coding:utf8
import torch as t
import time

class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path,change_opt=True):
        print path
        if self.opt.cuda==-1: data = t.load(path, map_location=lambda storage, loc: storage)
        else: data = t.load(path)
        if 'opt' in data:
            if change_opt: # change opt
                self.opt.parse(data['opt'],print_=False) # load origin opt file
                self.opt.embedding_path=None
                self.__init__(self.opt)
            self.load_state_dict(data['d']) # load para
        else:
            self.load_state_dict(data)
        return self

    def save(self, name=None,new=False):
        prefix = 'checkpoints/' + self.model_name + '_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix+name

        if new:
            data = {'opt':self.opt.state_dict(),'d':self.state_dict()} # if new, the save opt and d(parameters)
        else:
            data=self.state_dict()

        t.save(data, path)
        return path

    # lr1. 0.001, l2=0,???
    def get_optimizer(self,lr1,lr2=0,weight_decay = 0):
        ignored_params = list(map(id, self.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                        self.parameters())
        if lr2 is None: lr2 = lr1*0.5 
        optimizer = t.optim.Adam([
                dict(params=base_params,weight_decay = weight_decay,lr=lr1),
                {'params': self.parameters(), 'lr': lr2}
            ])
        return optimizer
