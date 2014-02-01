#import extra packages
import numpy
import sqlobject
from sqlobject import SQLObject, PickleCol, StringCol, IntCol, FloatCol, BoolCol, TimestampCol, ForeignKey, RelatedJoin, MultipleJoin, SingleJoin


class Test(SQLObject):
    """
    dataset and some meta data
    """

    organism = StringCol(default="")
    signal = PickleCol()
    
print "before"

sqlobject.sqlhub.processConnection = sqlobject.connectionForURI('mysql://cwidmer:mykyinva@10.37.40.83/multitask')
#sqlobject.sqlhub.processConnection = sqlobject.connectionForURI('mysql://root@localhost/multitask')
#sqlobject.sqlhub.processConnection = sqlobject.connectionForURI('sqlite:///fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/db/test.db')

print "after"
#Test.createTable()

a = numpy.ones((80, 80))

d = Test(organism="affe", signal=a)

print d

print "======================="

id = d.id

d2 = Test.get(id)

print d2

