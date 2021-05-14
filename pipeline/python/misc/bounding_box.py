# [zmin, zmax)
class BoundingBox:
    def __init__(self, zmin=None, zmax=None, ymin=None, ymax=None, xmin=None, xmax=None):
        self.zmin, self.zmax, self.ymin, self.ymax, self.xmin, self.xmax = zmin, zmax, ymin, ymax, xmin, xmax

    @classmethod
    def from_other(cls, other):
        assert isinstance(other, BoundingBox)
        return cls(zmin=other.zmin, zmax=other.zmax, ymin=other.ymin, ymax=other.ymax, xmin=other.xmin, xmax=other.xmax)

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            raise ValueError('does not know how to compare BoundingBox with {}'.format(other.__class__))
        return self.zmin == other.zmin and self.zmax == other.zmax and self.ymin == other.ymin and self.ymax == other.ymax and \
               self.xmin == other.xmin and self.xmax == other.xmax

    def empty(self):
        if any(bound is None for bound in [self.zmin, self.zmax, self.ymin, self.ymax, self.xmin, self.xmax]):
            return True
        if self.zmin >= self.zmax or self.ymin >= self.ymax or self.xmin >= self.xmax:
            return True

    def area(self):
        if self.empty():
            return 0
        return (self.zmax - self.zmin) * (self.ymax - self.ymin) * (self.xmax - self.xmin)

    # return smallest boundingbox that will enclose both self and other
    def enclose(self, other):
        assert isinstance(other, BoundingBox)
        if self.empty() and other.empty():
            return BoundingBox()
        elif self.empty():
            return BoundingBox.from_other(other)
        elif other.empty():
            return BoundingBox.from_other(self)
        else:
            zmax = max(self.zmax, other.zmax)
            zmin = min(self.zmin, other.zmin)
            ymax = max(self.ymax, other.ymax)
            ymin = min(self.ymin, other.ymin)
            xmax = max(self.xmax, other.xmax)
            xmin = min(self.xmin, other.xmin)
            return BoundingBox(zmin=zmin, zmax=zmax, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)

    def intersect(self, other):
        assert isinstance(other, BoundingBox)
        if self.empty() or other.empty():
            return BoundingBox()
        low_zbox = self if self.zmin <= other.zmin else other
        if abs(self.zmin - other.zmin) >= low_zbox.zmax - low_zbox.zmin:
            return BoundingBox()
        low_ybox = self if self.ymin <= other.ymin else other
        if abs(self.ymin - other.ymin) >= low_ybox.ymax - low_ybox.ymin:
            return BoundingBox()
        low_xbox = self if self.xmin <= other.xmin else other
        if abs(self.xmin - other.xmin) >= low_xbox.xmax - low_xbox.xmin:
            return BoundingBox()
        return BoundingBox(zmin=max(self.zmin, other.zmin), zmax=min(self.zmax, other.zmax),
                           ymin=max(self.ymin, other.ymin), ymax=min(self.ymax, other.ymax),
                           xmin=max(self.xmin, other.xmin), xmax=min(self.xmax, other.xmax))

    def __str__(self):
        return '{} {} {} {} {} {}'.format(self.zmin, self.zmax, self.ymin, self.ymax, self.xmin, self.xmax)


