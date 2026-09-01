using System;
using System.IO;
using System.IO.Compression;
using System.Text;

public static class UnzipShim
{
    public static int Main(string[] args)
    {
        try
        {
            if (args.Length < 2 || (args[0] != "-Z1" && args[0] != "-p"))
            {
                Console.Error.WriteLine("usage: unzip -Z1 archive | unzip -p archive entry");
                return 2;
            }

            using (var archive = ZipFile.OpenRead(args[1]))
            {
                if (args[0] == "-Z1")
                {
                    foreach (var entry in archive.Entries)
                        Console.WriteLine(entry.FullName.Replace('\\', '/'));
                    return 0;
                }

                if (args.Length < 3)
                    return 2;

                var wanted = args[2].Replace('\\', '/');
                foreach (var entry in archive.Entries)
                {
                    if (entry.FullName.Replace('\\', '/') != wanted)
                        continue;

                    using (var input = entry.Open())
                    using (var output = Console.OpenStandardOutput())
                        input.CopyTo(output);
                    return 0;
                }
            }

            Console.Error.WriteLine("missing zip entry: " + args[2]);
            return 1;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }
}
