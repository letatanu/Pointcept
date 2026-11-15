import copy
import importlib.util
import unittest
import torch
from .model import SerializedLocalCross,LearnedPooling
from .geometry import make_voxels,make_patches
from .layout import cross_layout,sample_cross
from .attention import PackedAttention,varlen


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        torch.set_num_threads(2)

    def model(self,local="none",cross="qk",checkpoint=False):
        return SerializedLocalCross(channels=(16,32),heads=(2,4),depths=(2,2),patch_sizes=(4,4),
            pool_sizes=(.4,),position_scales=(.4,.8),cross_group_size=2,cross_queries=(4,4),
            cross_keys=(2,2),chunk_size=3,cross_every=2,cross_dilations=(1,2),
            local_rotary=local,cross_rotary=cross,attention_backend="reference",checkpoint_blocks=checkpoint,
            serialization_backend="reference")

    def data(self):
        return dict(coord=torch.rand(39,3)*1.3,feat=torch.randn(39,4),offset=torch.tensor([23,39]))

    def test_dilation_partition_and_cloud_safety(self):
        xyz=torch.arange(68).float()[:,None].expand(-1,3)*.1
        batch=torch.tensor([0]*35+[1]*33)
        patches=make_patches(xyz,batch,.05,(0,1,2),4)
        layout=cross_layout(patches,batch,2,2,5,3)
        patchcloud=batch[patches.members[:,0]]
        # First cloud groups start with [0,2], [1,3], [4,6], [5,7], [8].
        self.assertEqual(layout.group[:9].tolist(),[0,1,0,1,2,3,2,3,4])
        for gid in range(layout.groups):
            self.assertEqual(len(patchcloud[layout.group==gid].unique()),1)
        for training in (False,True):
            q,k=sample_cross(patches,layout,2,training)
            qids,kids=q[q>=0],k[k>=0]
            self.assertEqual(len(qids),len(qids.unique()))
            self.assertEqual(len(kids),len(kids.unique()))
            self.assertTrue(((q>=0).sum(-1)<=5).all())
            self.assertTrue(((k>=0).sum(-1)<=6).all())
            for i in range(len(q)):
                self.assertEqual(len(batch[q[i][q[i]>=0]].unique()),1)
                self.assertEqual(int(batch[q[i][q[i]>=0]][0]),int(batch[k[i][k[i]>=0]][0]))

    def test_forward_backward_all_rotary_modes(self):
        for local,cross in (("none","none"),("none","qk"),("qk","none"),("qk","qk"),("qk","qkv")):
            model=self.model(local,cross).train()
            data=self.data()
            data["feat"].requires_grad_()
            output,debug=model(data,True)
            self.assertEqual(output.shape,(39,64))
            self.assertTrue(torch.isfinite(output).all())
            for record in debug:
                ids=record["local_members"]
                valid=ids[ids>=0]
                self.assertEqual(len(valid),len(valid.unique()))
                self.assertEqual(len(valid),len(record["batch"]))
            output.square().mean().backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,(local,cross,name))
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertTrue((data["feat"].grad.abs().sum(-1)>0).all())

    def test_eval_isolation_chunk_and_repeatability(self):
        model=self.model("qkv","qkv").eval()
        data=self.data()
        with torch.no_grad():
            output=model(data)
            torch.testing.assert_close(output,model(data),rtol=0,atol=0)
            for lo,hi in ((0,23),(23,39)):
                single=model(dict(coord=data["coord"][lo:hi],feat=data["feat"][lo:hi]))
                torch.testing.assert_close(output[lo:hi],single,rtol=2e-5,atol=3e-6)
            for module in model.modules():
                if isinstance(module,PackedAttention):
                    module.chunk_size=100
            torch.testing.assert_close(output,model(data),rtol=2e-5,atol=3e-6)

    def test_learned_pooling_weights_and_projection_shape(self):
        coord=torch.tensor([[0.,0,0],[.1,0,0],[.41,0,0]])
        cells=make_voxels(coord,torch.zeros(3,dtype=torch.long),.4)
        pool=LearnedPooling(8,16)
        feature=torch.randn(3,8,requires_grad=True)
        seen=[]
        handle=pool.project.register_forward_pre_hook(lambda module,args:seen.append(args[0].shape))
        out,_,_,_,weight=pool(feature,coord,torch.ones(3),cells,.4,True)
        handle.remove()
        self.assertEqual(seen,[torch.Size([2,8])])
        sums=torch.zeros(2).index_add(0,cells.inverse,weight)
        torch.testing.assert_close(sums,torch.ones(2))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertIsNotNone(pool.score[0].weight.grad)

    def test_singleton(self):
        output=self.model("qkv","qkv")(dict(coord=torch.zeros(1,3),feat=torch.randn(1,4)))
        self.assertEqual(output.shape,(1,64))
        output.mean().backward()
        self.assertTrue(torch.isfinite(output).all())

    def test_checkpoint_matches_loss_and_gradients(self):
        plain=self.model("qk","qkv").train()
        checked=copy.deepcopy(plain)
        checked.checkpoint_blocks=True
        data=self.data()
        torch.manual_seed(100)
        a=plain(data)
        a.square().mean().backward()
        torch.manual_seed(100)
        b=checked(data)
        b.square().mean().backward()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for (name,x),(_,y) in zip(plain.named_parameters(),checked.named_parameters()):
            torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=2e-6,msg=name)

    def test_qkv_inverse_rotation_relative_value_path(self):
        attention=PackedAttention(16,2,"qkv","reference",10,.25,2.).eval()
        feature=torch.randn(7,16)
        xyz=torch.randn(7,3)
        qtab=torch.tensor([[0,2],[4,6]])
        ktab=torch.tensor([[0,1,2],[4,5,6]])
        with torch.no_grad():
            a=attention(feature,xyz,qtab,ktab,2,3,1.)
            # Fixed feature tensors; only position translation is tested.
            b=attention(feature,xyz+torch.tensor([2.,-1.,3.]),qtab,ktab,2,3,1.)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_flash_amp_and_fp32_evaluation(self):
        if importlib.util.find_spec("flash_attn") is None:
            self.skipTest("flash-attn unavailable")
        model=self.model("qkv","qkv").cuda()
        for module in model.modules():
            if isinstance(module,PackedAttention):
                module.backend="flash_varlen"
        data={k:v.cuda() for k,v in self.data().items()}
        with torch.autocast("cuda",dtype=torch.float16):
            loss=model(data).float().square().mean()
        loss.backward()
        for name,p in model.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        model.eval()
        with torch.no_grad(),torch.autocast("cuda",enabled=False):
            output=model(data)
        self.assertTrue(torch.isfinite(output).all())


if __name__=="__main__":
    unittest.main()
