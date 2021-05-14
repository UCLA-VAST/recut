import time
import re


def time_stamp(output_format="%Y-%m-%d"):
    return time.strftime(output_format, time.localtime())


def current_dev_version():
    return 'dev_{}'.format(time_stamp('%Y-%m-%d-%H-%M-%S'))


# prior to v.1.0, deployed pipelines are versioned by a date time stamp
# symbolically assign these older pipeline v.0.8
def current_deploy_version():
    return 'v.3.1'


def current_version(deploy):
    return current_deploy_version() if deploy else current_dev_version()


def dated_deploy_version():
    return 'v.0.8'


# ubuntu 20.04 does not support : as file name
def dev_version_format():
    return '^dev_[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^dev_[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}$'


def deploy_version_format():
    return '^v\.[0-9]+\.[0-9]+$'


def version_format(deploy):
    return deploy_version_format() if deploy else dev_version_format()


# dates on which deployed versions, starting v.1.0 are packaged
# formatted as %Y-%m-%d
def deploy_version_dates():
    return {
        'v.0.8': '2020-02-01',
        'v.1.0': '2020-05-28',
        'v.1.1': '2020-06-10',
        'v.1.2': '2020-06-10',
        'v.2.0': '2020-07-27',
        'v.2.1': '2020-07-29',
        'v.2.2': '2020-08-28',
        'v.2.3': '2020-10-01',
        'v.3.0': '2020-12-14'
    }


class PipelineVersion:
    # if version string given, instantiate from version string
    # otherwise instantiate from current_version(deploy)
    # for version __eq__, __gt__, __lt__ comparisons, use following rule:
    # (1) if both dev version, compare using time stamp. if one dev and one deploy, retrieve deploy time stamp
    #     from deploy_version_dates() and compare using time stamp
    # (2) if both deploy version, compare using major and minor version numbers. multiple versioned deploy packages
    #     can be created on same day therefore can not be differentiated by date alone
    def __init__(self, version_str=None, deploy=True):
        if version_str is not None:
            # if version_str is not None, it should match dev or deploy format
            assert re.match(deploy_version_format(), version_str) or re.match(dev_version_format(), version_str)
            self.version_str = version_str
            self.deploy = re.match(deploy_version_format(), self.version_str) is not None
        else:
            self.deploy = deploy
            self.version_str = current_version(self.deploy)

    def __str__(self):
        return self.version_str

    def deploy_major_version(self):
        assert self.deploy
        return int(self.version_str.split('.')[1])

    def deploy_minor_version(self):
        assert self.deploy
        return int(self.version_str.split('.')[2])

    def dev_time_stamp(self):
        assert not self.deploy
        return self.version_str.replace('dev_', '')

    def time_since_epoch(self):
        if self.deploy:
            ts = time.strptime(deploy_version_dates()[self.version_str], '%Y-%m-%d')
        else:
            ts = time.strptime(self.dev_time_stamp(), '%Y-%m-%d-%H:%M:%S')
        return time.mktime(ts)

    def __eq__(self, other):
        assert isinstance(other, PipelineVersion)
        if self.deploy and other.deploy:
            return self.deploy_major_version() == other.deploy_major_version() and self.deploy_minor_version() == other.deploy_minor_version()
        else:
            return self.time_since_epoch() == other.time_since_epoch()

    def __gt__(self, other):
        assert isinstance(other, PipelineVersion)
        if self.deploy and other.deploy:
            if self.deploy_major_version() > other.deploy_major_version():
                return True
            elif self.deploy_major_version() == other.deploy_major_version():
                return self.deploy_minor_version() > other.deploy_minor_version()
            else:
                return False
        else:
            return self.time_since_epoch() > other.time_since_epoch()

    def __ne__(self, other):
        return not self == other

    def __ge__(self, other):
        return self == other or self > other

    def __lt__(self, other):
        return not self >= other

    def __le__(self, other):
        return not self > other






