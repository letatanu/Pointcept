import unittest
import torch
from .serialization import ORDERS,reference_encode,serialized_order
from .geometry import make_patches


class SerializationTests(unittest.TestCase):
    def test_morton_bit_convention_and_transpose(self):
        grid=torch.tensor([[0,0,0],[0,0,1],[0,1,0],[1,0,0]])
        torch.testing.assert_close(reference_encode(grid,1,"z"),torch.tensor([0,1,2,4]))
        for order in ("z","hilbert"):
            torch.testing.assert_close(reference_encode(grid,2,order+"-trans"),
                                       reference_encode(grid[:,[1,0,2]],2,order))

    def test_hilbert_small_cube_codes_and_continuity(self):
        cube=torch.cartesian_prod(torch.arange(2),torch.arange(2),torch.arange(2))
        torch.testing.assert_close(reference_encode(cube,1,"hilbert"),torch.tensor([0,1,3,2,7,6,4,5]))
        for depth in (2,3):
            cube=torch.cartesian_prod(*[torch.arange(2**depth)]*3)
            codes=reference_encode(cube,depth,"hilbert")
            torch.testing.assert_close(codes.sort().values,torch.arange(len(cube)))
            path=cube[codes.argsort()]
            torch.testing.assert_close((path[1:]-path[:-1]).abs().sum(-1),torch.ones(len(cube)-1,dtype=torch.long))

    def test_all_orders_cover_points_and_separate_clouds(self):
        torch.manual_seed(31)
        coord=torch.rand(27,3)*4
        coord[1]=coord[0]  # duplicate-grid ties remain stable
        batch=torch.tensor([0]*16+[1]*11)
        for order in ORDERS:
            indices=serialized_order(coord,batch,.05,order,"reference")
            torch.testing.assert_close(indices.sort().values,torch.arange(len(coord)))
            self.assertTrue((batch[indices][1:]>=batch[indices][:-1]).all())
            patches=make_patches(coord,batch,.05,order,5,"reference")
            valid=patches.members[patches.valid]
            torch.testing.assert_close(valid.sort().values,torch.arange(len(coord)))
            for row in patches.members:
                self.assertEqual(len(batch[row[row>=0]].unique()),1)

    def test_pointcept_parity_when_available(self):
        try:
            from pointcept.models.utils.serialization import encode
        except ImportError:
            self.skipTest("Pointcept serializer unavailable")
        torch.manual_seed(36)
        grid=torch.randint(0,2**8,(200,3))
        for order in ORDERS:
            expected=encode(grid,depth=8,order=order)
            torch.testing.assert_close(reference_encode(grid,8,order),expected.reshape(-1))

    def test_unknown_order_and_overflow(self):
        batch=torch.zeros(1,dtype=torch.long)
        with self.assertRaises(ValueError):
            serialized_order(torch.zeros(1,3),batch,1.,"unknown","reference")
        with self.assertRaises(ValueError):
            serialized_order(torch.ones(1,3)*100000,batch,1.,"hilbert","reference")
