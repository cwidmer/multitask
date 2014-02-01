#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009-2011 Christian Widmer
# Copyright (C) 2009-2011 Max-Planck-Society

"""
Created on 09.03.2009
@author: Christian Widmer
@summary: Establishes the database connection

"""

import sqlobject

sqlobject.sqlhub.processConnection = sqlobject.connectionForURI('mysql://cwidmer:mykyinva@huangho3/multitask')
